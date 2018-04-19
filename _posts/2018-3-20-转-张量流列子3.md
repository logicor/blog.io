
# 转 张量流例子3

<h1 id="版本说明"><a name="t0"></a>版本说明</h1>

<p>———-这次我不会忘记要写版本了分割线~</p>

<p>python：python3.5 <br>
tensorflow：tensorflow-0.12.1 <br>
numpy+mkl：numpy-1.11.3+mkl <br>
matplotlib：matplotlib-2.0.0 <br>
sklearn：scikit_learn-0.18.1 <br>
scipy：scipy-0.19.0 <br>
注：虽然代码里没有直接使用scipy和mkl，但是是sklearn的依赖，也是要下载安装好的。</p>



<h1 id="导入的包"><a name="t1"></a>导入的包</h1>

<p><img src="https://img-blog.csdn.net/20170408222351714?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""></p>



<h1 id="数据源"><a name="t2"></a>数据源</h1>

<p>本次实验用到的数据源是网上下载的<strong>哈利波特1-7</strong>，经过去符号处理，只留下了单词序列</p>



<h1 id="代码"><a name="t3"></a>代码</h1>

<p>已经上传到Github上了。 <br>
<a href="https://github.com/LouisScorpio/datamining/tree/master/tensorflow-program/nlp/word2vec" target="_blank">word2vec_harrypotter</a></p>



<h1 id="结果"><a name="t4"></a>结果</h1>

<p>嗯哼，做了word2vec有什么效果呢？ <br>
效果在这里 <br>
<img src="https://img-blog.csdn.net/20170408233548407?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""></p>

<p>实验取了单词频数top150成图，这里截取了一部分。 <br>
可以看到，黄色圈出来的back,out,behind,into,up距离比较近，himself,her,him,them,us聚在一起，还有for,though,than,but,and这些聚在一起。 <br>
当然，迭代次数多一些，效果可能会更棒。</p>



<h1 id="说明"><a name="t5"></a>说明</h1>

<p>代码参考了<strong>Tensorflow实战   黄文坚 唐源著</strong></p>                