# taobao_fresh
 主要代码都在 item_id 分支中
 
 Src\LR代码已经废弃
 
 Src\LR-hit LogisticRegression 算法
 Src\RF RandomForest 算法
 Src\feature 生成特征矩阵
 Src\GBDT GBDT 算法
 Src\samples  样本选择
 src\taobao_fresh_comp.py 代码入口， 由 taobao_fresh_comp_controller.py 启动
 src\taobao_fresh_comp_controller.py 主控代码，启动若干个子进程调用taobao_fresh_comp.py，等待子进程结束后将生成的预测结果合成
 src\common.py 公用函数
 其他代码暂时没有用到
 
 运行方式：
 python taobao_fresh_comp_controller.py， 在其中会启动子进程：
 python taobao_fresh_comp.py start_from=%d user_cnt=%d slide=%d topk=%d test=0 min_proba=0.5
 start_from： 表示从第几个用户开始读取记录
 user_cnt: 本次运行需要加载多少个用户的数据
 slide: 滑动窗口大小
 topk： 按照预测的概率由高到低取topk 个结果
 test： 0 模型最后会走verify流程，计算F1, 1: 不走verify流程，输出预测结果
 min_proba: 预测可能会购买的最小概率
 
 例如：
 python taobao_fresh_comp.py start_from=0 user_cnt=2000 slide=5 topk=1000 test=0 min_proba=0.5
 

 
