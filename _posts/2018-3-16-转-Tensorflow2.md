
#转 张量流例子2

<p>实验用到的数据长这个样子： <br>
<img src="https://img-blog.csdn.net/20170225002322179?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""></p>

<p>label是标签y，也就是下一日的最高价。列C——I为输入特征。 <br>
本实例用前5800个数据做训练数据。</p>

<p>单因素输入特征及RNN、LSTM的介绍请戳上一篇 <a href="http://blog.csdn.net/mylove0414/article/details/55805974" target="_blank">Tensorflow实例：利用LSTM预测股票每日最高价（一）</a></p>



<h1 id="导入包及声明常量"><a name="t0"></a>导入包及声明常量</h1>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-keyword">import</span> pandas <span class="hljs-keyword">as</span> pd
<span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-comment">#定义常量</span>
rnn_unit=<span class="hljs-number">10</span>       <span class="hljs-comment">#hidden layer units</span>
input_size=<span class="hljs-number">7</span>      
output_size=<span class="hljs-number">1</span>
lr=<span class="hljs-number">0.0006</span>         <span class="hljs-comment">#学习率</span></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li></ul></pre>



<h1 id="导入数据"><a name="t1"></a>导入数据</h1>



<pre class="prettyprint"><code class="language-python hljs  has-numbering">f=open(<span class="hljs-string">'dataset.csv'</span>) 
df=pd.read_csv(f)     <span class="hljs-comment">#读入股票数据</span>
data=df.iloc[:,<span class="hljs-number">2</span>:<span class="hljs-number">10</span>].values   <span class="hljs-comment">#取第3-10列</span></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li></ul></pre>



<h1 id="生成训练集测试集"><a name="t2"></a>生成训练集、测试集</h1>

<p>考虑到真实的训练环境，这里把每批次训练样本数（batch_size）、时间步（time_step）、训练集的数量（train_begin,train_end）设定为参数，使得训练更加机动。</p>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-comment">#——————————获取训练集——————————</span>
<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">get_train_data</span><span class="hljs-params">(batch_size=<span class="hljs-number">60</span>,time_step=<span class="hljs-number">20</span>,train_begin=<span class="hljs-number">0</span>,train_end=<span class="hljs-number">5800</span>)</span>:</span>
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=<span class="hljs-number">0</span>))/np.std(data_train,axis=<span class="hljs-number">0</span>)  <span class="hljs-comment">#标准化</span>
    train_x,train_y=[],[]   <span class="hljs-comment">#训练集x和y初定义</span>
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(len(normalized_train_data)-time_step):
       <span class="hljs-keyword">if</span> i % batch_size==<span class="hljs-number">0</span>:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:<span class="hljs-number">7</span>]
       y=normalized_train_data[i:i+time_step,<span class="hljs-number">7</span>,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    <span class="hljs-keyword">return</span> batch_index,train_x,train_y

<span class="hljs-comment">#——————————获取测试集——————————</span>
<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">get_test_data</span><span class="hljs-params">(time_step=<span class="hljs-number">20</span>,test_begin=<span class="hljs-number">5800</span>)</span>:</span>
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=<span class="hljs-number">0</span>)
    std=np.std(data_test,axis=<span class="hljs-number">0</span>)
    normalized_test_data=(data_test-mean)/std  <span class="hljs-comment">#标准化</span>
    size=(len(normalized_test_data)+time_step-<span class="hljs-number">1</span>)//time_step  <span class="hljs-comment">#有size个sample </span>
    test_x,test_y=[],[]  
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(size-<span class="hljs-number">1</span>):
       x=normalized_test_data[i*time_step:(i+<span class="hljs-number">1</span>)*time_step,:<span class="hljs-number">7</span>]
       y=normalized_test_data[i*time_step:(i+<span class="hljs-number">1</span>)*time_step,<span class="hljs-number">7</span>]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+<span class="hljs-number">1</span>)*time_step:,:<span class="hljs-number">7</span>]).tolist())
    test_y.extend((normalized_test_data[(i+<span class="hljs-number">1</span>)*time_step:,<span class="hljs-number">7</span>]).tolist())
    <span class="hljs-keyword">return</span> mean,std,test_x,test_y</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li><li style="color: rgb(153, 153, 153);">19</li><li style="color: rgb(153, 153, 153);">20</li><li style="color: rgb(153, 153, 153);">21</li><li style="color: rgb(153, 153, 153);">22</li><li style="color: rgb(153, 153, 153);">23</li><li style="color: rgb(153, 153, 153);">24</li><li style="color: rgb(153, 153, 153);">25</li><li style="color: rgb(153, 153, 153);">26</li><li style="color: rgb(153, 153, 153);">27</li><li style="color: rgb(153, 153, 153);">28</li><li style="color: rgb(153, 153, 153);">29</li><li style="color: rgb(153, 153, 153);">30</li><li style="color: rgb(153, 153, 153);">31</li><li style="color: rgb(153, 153, 153);">32</li></ul></pre>



<h1 id="构建神经网络"><a name="t3"></a>构建神经网络</h1>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-comment">#——————————————————定义神经网络变量——————————————————</span>
<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">lstm</span><span class="hljs-params">(X)</span>:</span>     
    batch_size=tf.shape(X)[<span class="hljs-number">0</span>]
    time_step=tf.shape(X)[<span class="hljs-number">1</span>]
    w_in=weights[<span class="hljs-string">'in'</span>]
    b_in=biases[<span class="hljs-string">'in'</span>]  
    input=tf.reshape(X,[-<span class="hljs-number">1</span>,input_size])  <span class="hljs-comment">#需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入</span>
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-<span class="hljs-number">1</span>,time_step,rnn_unit])  <span class="hljs-comment">#将tensor转成3维，作为lstm cell的输入</span>
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  <span class="hljs-comment">#output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果</span>
    output=tf.reshape(output_rnn,[-<span class="hljs-number">1</span>,rnn_unit]) <span class="hljs-comment">#作为输出层的输入</span>
    w_out=weights[<span class="hljs-string">'out'</span>]
    b_out=biases[<span class="hljs-string">'out'</span>]
    pred=tf.matmul(output,w_out)+b_out
    <span class="hljs-keyword">return</span> pred,final_states
</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li></ul></pre>



<h1 id="训练模型"><a name="t4"></a>训练模型</h1>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-comment">#——————————————————训练模型——————————————————</span>
<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">train_lstm</span><span class="hljs-params">(batch_size=<span class="hljs-number">80</span>,time_step=<span class="hljs-number">15</span>,train_begin=<span class="hljs-number">0</span>,train_end=<span class="hljs-number">5800</span>)</span>:</span>
    X=tf.placeholder(tf.float32, shape=[<span class="hljs-keyword">None</span>,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[<span class="hljs-keyword">None</span>,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    <span class="hljs-comment">#损失函数</span>
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-<span class="hljs-number">1</span>])-tf.reshape(Y, [-<span class="hljs-number">1</span>])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=<span class="hljs-number">15</span>)
    module_file = tf.train.latest_checkpoint()    
    <span class="hljs-keyword">with</span> tf.Session() <span class="hljs-keyword">as</span> sess:
        <span class="hljs-comment">#sess.run(tf.global_variables_initializer())</span>
        saver.restore(sess, module_file)
        <span class="hljs-comment">#重复训练2000次</span>
        <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">2000</span>):
            <span class="hljs-keyword">for</span> step <span class="hljs-keyword">in</span> range(len(batch_index)-<span class="hljs-number">1</span>):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+<span class="hljs-number">1</span>]],Y:train_y[batch_index[step]:batch_index[step+<span class="hljs-number">1</span>]]})
            print(i,loss_)
            <span class="hljs-keyword">if</span> i % <span class="hljs-number">200</span>==<span class="hljs-number">0</span>:
                print(<span class="hljs-string">"保存模型："</span>,saver.save(sess,<span class="hljs-string">'stock2.model'</span>,global_step=i))</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li><li style="color: rgb(153, 153, 153);">19</li><li style="color: rgb(153, 153, 153);">20</li><li style="color: rgb(153, 153, 153);">21</li></ul></pre>

<p>嗯，这里说明一下，这里的参数是基于已有模型恢复的参数，意思就是说之前训练过模型，保存过神经网络的参数，现在再取出来作为初始化参数接着训练。如果是第一次训练，就用sess.run(tf.global_variables_initializer())，也就不要用到 module_file = tf.train.latest_checkpoint() 和saver.store(sess, module_file)了。</p>



<h1 id="测试"><a name="t5"></a>测试</h1>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-comment">#————————————————预测模型————————————————————</span>
<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">prediction</span><span class="hljs-params">(time_step=<span class="hljs-number">20</span>)</span>:</span>
    X=tf.placeholder(tf.float32, shape=[<span class="hljs-keyword">None</span>,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    <span class="hljs-keyword">with</span> tf.Session() <span class="hljs-keyword">as</span> sess:
        <span class="hljs-comment">#参数恢复</span>
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file) 
        test_predict=[]
        <span class="hljs-keyword">for</span> step <span class="hljs-keyword">in</span> range(len(test_x)-<span class="hljs-number">1</span>):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-<span class="hljs-number">1</span>))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[<span class="hljs-number">7</span>]+mean[<span class="hljs-number">7</span>]
        test_predict=np.array(test_predict)*std[<span class="hljs-number">7</span>]+mean[<span class="hljs-number">7</span>]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) <span class="hljs-comment">#acc为测试集偏差</span></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li></ul></pre>

<p>最后的结果画出来是这个样子：</p>

<p><img src="https://img-blog.csdn.net/20170225003402417?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""> <br>
红色折线是真实值，蓝色折线是预测值</p>

<p>偏差大概在1.36%</p>

<p>代码和数据上传到了github上，想要的戳<a href="https://github.com/LouisScorpio/datamining/blob/master/tensorflow-program/rnn/stock_predict/stock_predict_2.py" target="_blank">全部代码</a></p>

<p>注！：如要转载，请经过本人允许并注明出处！</p>                