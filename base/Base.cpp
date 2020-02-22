#include <windows.h>
#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <process.h>


extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;    // 线程id 
	INT *batch_h; // 三元组 head 
	INT *batch_t; // 三元组 tail 
	INT *batch_r; // 三元组 relation 
	REAL *batch_y; // 是否为正例 1 正；-1 负 
	INT batchSize; // 未知 猜测为 批大小 
	INT negRate; 	// 负实体例比率 ，每一个正例  negRate个替换h或者t的负例 
	INT negRelRate;	// 负关系比率 
	bool p;			// 未知 
	bool val_loss;	// 用于评估损失。即 不含负例 
	INT mode;		// 0：一半随机h，一半随机t；-1:sampling_head； 1：sampling_tail 
	bool filter_flag;	// 滤假负例 
};

// 线程执行  batch_h[i] 是正例 batch_h[i+ batchSize] 是负例， batch_h[i+negRate*batchSize]也是负例 
unsigned int __stdcall getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
//	printf("this is thread #%lld, its addr: %lld, id addr: %lld\n",para->id,para,&(para->id));
//	printf("print more para:%lld %lld %lld %lld %lld %lld \n",para -> batch_h,para -> batch_t,para -> batch_r,para -> batchSize,para -> negRate,para -> negRelRate);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
	INT lef, rig;
	//分配每个线程的任务 
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads); //从 0 开始 
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1); // 每各线程，多做一个，以防不足。  
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	if (val_loss == false) { //如果 需要负例 
		for (INT batch = lef; batch < rig; batch++) { 
			INT i = rand_max(id, trainTotal); // 随机得到一个 三元组 的 编号[0,trainTotal) 
			batch_h[batch] = trainList[i].h;
			batch_t[batch] = trainList[i].t;
			batch_r[batch] = trainList[i].r;
			batch_y[batch] = 1;  // 标明是正例 
//			printf("data #%lld: it's #%lld thiple,h is #%lld\n",batch,i,batch_h[batch]);
			INT last = batchSize;
			
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){ // 负例 对半随机 
					if (bernFlag)  //默认等于零 
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) { //随机 替换h 或者 t 
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				} else {
					if(mode == -1){ //负例 随机h 
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					} else { //负例 随机t 
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				}
			}
			// 
			for (INT times = 0; times < negRelRate; times++) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				batch_y[batch + last] = -1;
				last += batchSize;
			}
		}
	}
	else{ //不含负例 
		for (INT batch = lef; batch < rig; batch++){
			batch_h[batch] = validList[batch].h;
			batch_t[batch] = validList[batch].t;
			batch_r[batch] = validList[batch].r;
			batch_y[batch] = 1;
		}
	}
//	pthread_exit(NULL);
	_endthreadex(0);
	return 0;
}


// 启用多线程 生成 一批数据 
extern "C"
void sampling(
		INT *batch_h, 
		INT *batch_t, 
		INT *batch_r, 
		REAL *batch_y, 
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	
	HANDLE  *pt = (HANDLE  *)malloc(workThreads * sizeof(HANDLE ));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
//		printf("in sampling: threads#%lld para's addr: %lld, id addr: %lld\n",para[threads].id,para+threads,&((para+threads)->id));
		pt[threads] = (HANDLE)_beginthreadex(NULL,0,getBatch,(void*)(para+threads),0,NULL);
//		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}

//	for (INT threads = 0; threads < workThreads; threads++)
//		pthread_join(pt[threads], NULL); 
	WaitForMultipleObjects(workThreads, pt, TRUE, INFINITE);
	free(pt);
	free(para);
}
void foo();
int main() {
	// test sampling function 
//	setInPath("D:/py_workspace/machine_learning/experiment/paper/OpenKE/benchmarks/FB15K237/");
//	randReset();
//	setWorkThreads(2);
//	printf("%s\n",inPath);
	importTrainFiles();
//	INT a[100],b[100],c[100];
//	REAL d[100];
//	sampling(a,b,c,d,16,2,0,0,0,0,0);
//	scanf("%d",&pos);

	// test importTestFiles function 
//	// needed envirement: setInPath、randReset、importTestFiles 
//	setInPath("D:/py_workspace/machine_learning/experiment/paper/OpenKE/benchmarks/FB15K237/");
//	randReset();
//	importTestFiles();
//	importTypeFiles();
	
	return 0;
}
