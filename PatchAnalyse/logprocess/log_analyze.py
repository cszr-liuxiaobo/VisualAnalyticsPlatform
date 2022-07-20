import os

import numpy as np
import re
def get_attri_lists(path):
    with open(path,'r',encoding='utf-8')as f:
        lines = f.readlines()
        print(lines)
        LearningRates=[]
        testbatavgloss=[]
        TestAccuracies = []
        TrainAccuracies = []
        for i in range(0,len(lines)):
            if 'LearningRate' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                # print(elements[-1])
                LearningRates.append(float(elements_number[0]))
            elif 'testbatavgloss' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                testbatavgloss.append(float(elements_number[0]))
            elif 'TestAccuracy' in lines[i]:
                elements = lines[i].split(':')
                # print(elements[-1])
                TestAccuracies.append(elements[-1])
            elif 'TrainAccuracy' in lines[i]:
                elements = lines[i].split(':')
                # print(elements[-1])
                TrainAccuracies.append(elements[-1])
    return LearningRates,testbatavgloss,TestAccuracies,TrainAccuracies


def data_process(path):
    LearningRates,batavglosses,TestAccuracies,TrainAccuracies=get_attri_lists(path)
    LR_max=max(LearningRates)
    LR_min=min(LearningRates)
    avg_loss_max = max(batavglosses)
    avg_loss_min=min(batavglosses)
    # avg_loss_median= np.median(batavglosses)
    avg_loss_last = batavglosses[-1]
    print("LR最大:{}\nLR最小:{}\navg_loss最大:{}\navg_loss最小:{}\navg_loss最后:{}"
          .format(LR_max,LR_min,avg_loss_max,avg_loss_min,avg_loss_last))
    TrainAccuracies_min = min(TrainAccuracies)
    TrainAccuracies_max=max(TrainAccuracies)
    TrainAccuracies_last = TrainAccuracies[-1]
    print("Train最小:{}Train最大:{}Train最后:{}"
          .format(TrainAccuracies_min,TrainAccuracies_max,TrainAccuracies_last))
    TestAccuracies_min = min(TestAccuracies)
    TestAccuracies_max=max(TestAccuracies)
    TestAccuracies_last = TestAccuracies[-1]
    print("Test最小:{}Test最大:{}Test最后:{}"
          .format(TestAccuracies_min,TestAccuracies_max,TestAccuracies_last))

# -------------------获取precision,recall,f1-score,accuracy相关数据---------------
def get_prfa_attri_lists(path):
    with open(path,'r',encoding='utf-8')as f:
        lines = f.readlines()
        print(lines)
        # 对日志中test相关的prfa进行记录
        test_Precision_normal=[]
        test_Precision_problem=[]
        test_Recall_normal=[]
        test_Recall_problem=[]
        test_f1_score_normal=[]
        test_f1_score_problem=[]

        # 对日志中train相关的prfa进行记录
        train_Precision_normal=[]
        train_Precision_problem=[]
        train_Recall_normal=[]
        train_Recall_problem=[]
        train_f1_score_normal=[]
        train_f1_score_problem=[]

        for i in range(0,len(lines)):
            if 'test_Precision' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                # print('elements_number',float(elements_number[0]))
                test_Precision_normal.append(float(elements_number[0]))
                test_Precision_problem.append(float(elements_number[1]))
            elif 'test_Recall' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                test_Recall_normal.append(float(elements_number[0]))
                test_Recall_problem.append(float(elements_number[1]))
            elif 'test_f1_score' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                test_f1_score_normal.append(float(elements_number[0]))
                test_f1_score_problem.append(float(elements_number[1]))


            if 'train_Precision' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                train_Precision_normal.append(float(elements_number[0]))
                train_Precision_problem.append(float(elements_number[1]))
            elif 'train_Recall' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                train_Recall_normal.append(float(elements_number[0]))
                train_Recall_problem.append(float(elements_number[1]))
            elif 'train_f1_score' in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                train_f1_score_normal.append(float(elements_number[0]))
                train_f1_score_problem.append(float(elements_number[1]))
    return test_Precision_normal,test_Precision_problem,\
           test_Recall_normal,test_Recall_problem,test_f1_score_normal,\
           test_f1_score_problem,train_Precision_normal,train_Precision_problem,\
           train_Recall_normal,train_Recall_problem,train_f1_score_normal,train_f1_score_problem

def prfa_data_processing(path):
    test_Precision_normal, test_Precision_problem, \
    test_Recall_normal, test_Recall_problem, \
    test_f1_score_normal, test_f1_score_problem, \
    train_Precision_normal, train_Precision_problem, \
    train_Recall_normal, train_Recall_problem, train_f1_score_normal, train_f1_score_problem=get_prfa_attri_lists(path)

    test_f1_score_normal_max=max(test_f1_score_normal)
    test_f1_score_normal_min=min(test_f1_score_normal)
    test_f1_score_normal_last=test_f1_score_normal[-1]
    print("Test_f1_score_normal最小:{}\nTest_f1_score_normal最大:{}\nTest_f1_score_normal最后:{}\n"
          .format(test_f1_score_normal_min,test_f1_score_normal_max,test_f1_score_normal_last))

    test_f1_score_problem_max=max(test_f1_score_problem)
    test_f1_score_problem_min=min(test_f1_score_problem)
    test_f1_score_problem_last=test_f1_score_problem[-1]
    print("Test_f1_score_problem最小:{}\nTest_f1_score_problem最大:{}\nTest_f1_score_problem最后:{}\n"
          .format(test_f1_score_problem_min,test_f1_score_problem_max,test_f1_score_problem_last))

# --------------------获取最后一轮关于filter的precision,recall,f1-score,accuracy相关数据-------------
def get_data_prfa_attri_lists(path,filter):
    # 对日志中test相关的prfa进行记录
    filter_Precision_normal = []
    filter_Precision_problem = []
    filter_Recall_normal = []
    filter_Recall_problem = []
    filter_f1_score_normal = []
    filter_f1_score_problem = []
    filterAccuracies=[]

    with open(path,'r',encoding='utf-8')as f:
        lines = f.readlines()
        print(lines)

        for i in range(0,len(lines)):
            if '{}_Precision'.format(filter) in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                # print('elements_number',float(elements_number[0]))
                filter_Precision_normal.append(float(elements_number[0]))
                filter_Precision_problem.append(float(elements_number[1]))
            elif '{}_Recall'.format(filter) in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                filter_Recall_normal.append(float(elements_number[0]))
                filter_Recall_problem.append(float(elements_number[1]))
            elif '{}_f1_score'.format(filter) in lines[i]:
                elements = lines[i].split(':')
                elements_number = re.findall('\d+(?:\.\d+)?', elements[-1])
                filter_f1_score_normal.append(float(elements_number[0]))
                filter_f1_score_problem.append(float(elements_number[1]))
            elif "{}_accuracy_score".format(filter) in lines[i]:
                elements = lines[i].split(':')
                filterAccuracies.append(elements[-1])



    return filter_Precision_normal,filter_Precision_problem,\
           filter_Recall_normal,filter_Recall_problem,filter_f1_score_normal,\
           filter_f1_score_problem,filterAccuracies

def filters_prfa_data_processing(path,filter):
    filter_Precision_normal, filter_Precision_problem, \
    filter_Recall_normal, filter_Recall_problem, filter_f1_score_normal, \
    filter_f1_score_problem,filterAccuracies=get_data_prfa_attri_lists(path,filter)

    f1_score_normal=filter_f1_score_normal
    print("{}_f1_score_normal:{}"
          .format(filter,f1_score_normal))

    f1_score_problem=filter_f1_score_problem
    print("{}_f1_score_problem:{}"
          .format(filter,f1_score_problem))

    meanfilter_accuracy_score=filterAccuracies[-1]
    print("{}_accuracy_score:{}"
          .format(filter,meanfilter_accuracy_score))

if __name__ == '__main__':
    current_path = os.path.split(os.path.realpath(__file__))[0]
    father_path=os.path.dirname(current_path)
    print(father_path)
    log_path = os.path.join(os.path.join(father_path, 'logprocess'), 'logs') # 存放log文件的路径
    path = os.path.join(log_path,'result_experiment5.log')

    data_process(path)
    prfa_data_processing(path)

    # 获得filter的相关数据
    print("------------filterssssss-------------------")
    print("------------filterssssss-------------------")
    print("------------filterssssss-------------------")
    for filter in ['nofilter','meanfilter3','meanfilter5','meanfilter7','meanfilter9','meanfilter11','meanfilter13']:
        filters_prfa_data_processing(path,filter=filter)
