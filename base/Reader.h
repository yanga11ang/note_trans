#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <unistd.h>
#include <algorithm>
//#include <iostream> //ctype 不支持 
#include <cmath>
#include <typeinfo.h >
INT *freqRel, *freqEnt; // 记录 rel 和 ent 出现的次数  
INT *lefHead, *rigHead; // 记录以h,r,t排序，每个h在数组第一次和最后一次出现的位置 
INT *lefTail, *rigTail; // 记录以t,r,h排序，每个t在数组第一次和最后一次出现的位置  
INT *lefRel, *rigRel;	// 记录以h,t,r排序，每个h在数组第一次和最后一次出现的位置
REAL *left_mean, *right_mean; // left_mean 平均每个关系，有多少个h实体 或 t实体
REAL *prob;
Triple *trainList; // 保存所有的正例 
Triple *trainHead; // 正例以h,r,t排序，
Triple *trainTail; // 正例以t,r,h排序，
Triple *trainRel;  // 正例以h,t,r排序

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
	INT inPath_len = strlen(inPath);
    fin = fopen(strcat(inPath,"kl_prob.txt"), "r");
    inPath[inPath_len]='\0';
    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	
	FILE *fin;
	INT tmp;
	INT inPath_len = strlen(inPath); 
	fin = fopen(strcat(inPath , "relation2id.txt"), "r");
	inPath[inPath_len]='\0';
	tmp = fscanf(fin, "%lld", &relationTotal);
	printf("The total of relations is %lld.\n", relationTotal);
	fclose(fin);
	
	
	fin = fopen(strcat(inPath , "entity2id.txt"), "r");
	inPath[inPath_len]='\0';
	tmp = fscanf(fin, "%lld", &entityTotal);
	printf("The total of entities is %lld.\n", entityTotal);
	fclose(fin);

	fin = fopen(strcat(inPath , "train2id.txt"), "r");
	inPath[inPath_len]='\0';
	tmp = fscanf(fin, "%lld", &trainTotal);
	//根据数据集的大小 分配数组内存 
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	// 记录 rel 或者 ent 出现的次数 
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	//将测试集 读取到  trainList数组 
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%lld", &trainList[i].h);
		tmp = fscanf(fin, "%lld", &trainList[i].t);
		tmp = fscanf(fin, "%lld", &trainList[i].r);
	}
	fclose(fin);
	// 排序，从小到大 规则 依次是 h r t 
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
	
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	//去重复 
	for (INT i = 1; i < tmp; i++) // h不同 ，或r不同 或t不同 
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}
	// h, r, t排序 
	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	// t, r, h 排序 
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	// h, t, r 排序 
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);
	printf("The total of train triples is %lld.\n", trainTotal);
	
	// 分别记录不同排序方式，指定元素第一个和最后一个的未知 分配内存空间 
	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
	// 初始化为-1 
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);
	//得出 每个元素 在不同数组中的 第一次出现和最后一次出现位置 
	for (INT i = 1; i < trainTotal; i++) {
		// t, r, h 排序 
		if (trainTail[i].t != trainTail[i - 1].t) { //t 不相同 
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		// h, r, t排序
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		// h, t, r 排序, 结果应该和上一个相同 
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	// 补遗 数组第一个元素 和最后一个元素 
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;
	
	// relationTotal大小 分配两个数组 
	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	// 分析每种rel 的性质  
	// left_mean 存每种rel有多少种h 
	// right_mean 存 每种rel有多少种t 
	for (INT i = 0; i < entityTotal; i++) {
		// 枚举 h=i时，每一种 r+=1 
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		// h开头只有一个的情况 ，不应该有if;上述循环，没有考虑第一种rel 
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		// 枚举t = i 时，每种 r++ 
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	// left_mean 平均每个关系，有多少个h实体  
	// right_mean 平均每个关系，有多少个t实体  
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

Triple *testList;
Triple *validList;
Triple *tripleList;

// 读取测试集， 
extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;
    
    INT inPath_len= strlen(inPath);
    fin = fopen(strcat(inPath , "relation2id.txt"), "r");
    inPath[inPath_len]='\0';
    tmp = fscanf(fin, "%lld", &relationTotal);
    fclose(fin);

    fin = fopen(strcat(inPath , "entity2id.txt"), "r");
    inPath[inPath_len]='\0';
    tmp = fscanf(fin, "%lld", &entityTotal);
    fclose(fin);

    FILE* f_kb1 = fopen(strcat(inPath , "test2id.txt"), "r");
    inPath[inPath_len]='\0';
    FILE* f_kb2 = fopen(strcat(inPath , "train2id.txt"), "r");
    inPath[inPath_len]='\0';
    FILE* f_kb3 = fopen(strcat(inPath , "valid2id.txt"), "r");
    inPath[inPath_len]='\0';
    tmp = fscanf(f_kb1, "%lld", &testTotal);
    tmp = fscanf(f_kb2, "%lld", &trainTotal);
    tmp = fscanf(f_kb3, "%lld", &validTotal);
    
    tripleTotal = testTotal + trainTotal + validTotal;
    
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) { //读取测试集 三元组 
        tmp = fscanf(f_kb1, "%lld", &testList[i].h);
        tmp = fscanf(f_kb1, "%lld", &testList[i].t);
        tmp = fscanf(f_kb1, "%lld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) { // 读取训练集 三元组 
        tmp = fscanf(f_kb2, "%lld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%lld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%lld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) { // 读取验证集 三元组 
        tmp = fscanf(f_kb3, "%lld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%lld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%lld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);
	
	//  h r t
    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    // r h t 
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    // r h t
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %lld.\n", testTotal);
    printf("The total of valid triples is %lld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    
    // 找出 测试集上 每种关系的第一个位置 和最后一个位置 
    for (INT i = 1; i < testTotal; i++) {
		if (testList[i].r != testList[i-1].r) { // 前后不是一种关系 
		    testRig[testList[i-1].r] = i - 1;   
		    testLef[testList[i].r] = i;
		}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    // 找出 验证集上 每种关系的 第一个位置 和最后一个位置 
    for (INT i = 1; i < validTotal; i++) {
		if (validList[i].r != validList[i-1].r) {
		    validRig[validList[i-1].r] = i - 1;
		    validLef[validList[i].r] = i;
		}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

INT* head_lef; // relationTotal  每个关系 在head_type中出现的 第一个位置，
INT* head_rig; // relationTotal  每个关系 在head_type中出现的 最后一个位置
INT* tail_lef; // relationTotal  每个关系 在tail_type中出现的 第一个位置，
INT* tail_rig; // relationTotal  每个关系 在tail_type中出现的 最后一个位置
INT* head_type; // total_lef  所有在关系左边出现的 实体 
INT* tail_type; // total_rig 所有在关系右边出现的 实体 

extern "C"
void importTypeFiles() {
	
    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    INT inPath_len = strlen(inPath);
    FILE* f_type = fopen(strcat(inPath , "type_constrain.txt"),"r");
    inPath[inPath_len]='\0';
    INT tmp;
    // just calculate the number of total_lef and total_rig
    tmp = fscanf(f_type, "%lld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%lld %lld", &rel, &tot); // 关系id 和 constrain数量 
		for (INT j = 0; j < tot; j++) {
			tmp = fscanf(f_type, "%lld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%lld%lld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%lld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen(strcat(inPath , "type_constrain.txt"),"r");
    inPath[inPath_len]='\0';
    
    tmp = fscanf(f_type, "%lld", &tmp);
    printf("%lld\n",relationTotal);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%lld%lld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%lld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]); //排序，从小到大 
        tmp = fscanf(f_type, "%lld%lld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%lld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
