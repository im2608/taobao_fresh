# taobao_fresh
主要代码都在 item_id 分支中

Src\LR代码已经废弃

Src\LR-hit LogisticRegression 算法 Src\RF RandomForest 算法

运行方式： python taobao_fresh_comp.py

参数在代码内指定： common.py 中： 
algo = "RF" or algo = "LR" 可以指定使用哪种算法

taobao_fresh_comp.py 中： 
need_output = 1 表示输出预测结果， 输出到 \src..\output\中, 如果输出预测结果，则不会执行 verify 流程 
need_output = 0 表示验证预测结果

start_from = 0 表示从第几个用户开始读取记录 user_cnt = 0 表示读取多少个用户的记录 
通过这两个值，可以在用户集合中取得任意数量的用户来验证算法 
user_cnt = 0 表示取得所有的用户（一般在 need_output = 1 是使用），此时 start_from 会被忽略
