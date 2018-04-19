
#转 张量流实列1

<p>这一部分主要涉及循环神经网络的理论，讲的可能会比较简略。</p>



<h2 id="什么是rnn"><a name="t1"></a>什么是RNN</h2>

<p>RNN全称循环神经网络（Recurrent Neural Networks），是用来处理序列数据的。在传统的神经网络模型中，从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。但是这种普通的神经网络对于很多关于时间序列的问题却无能无力。例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的。RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面时刻的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。 <br>
说了这么多，用一张图表示，就是这个样子。</p>

<p><img src="https://img-blog.csdn.net/20170219172311425?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""> <br>
传统的神经网络中，数据从输入层输入，在隐藏层加工，从输出层输出。RNN不同的就是在隐藏层的加工方法不一样，后一个节点不仅受输入层输入的影响，还包受上一个节点的影响。 <br>
展开来就是这个样子： <br>
<img src="https://img-blog.csdn.net/20170219172026142?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""></p>

<p>图中的<span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-1-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>x</mi><mrow class=&quot;MJX-TeXAtom-ORD&quot;><mi>t</mi><mo>&amp;#x2212;</mo><mn>1</mn></mrow></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-1" style="width: 2.19em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.823em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1001.82em, 2.552em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-2"><span class="msubsup" id="MathJax-Span-3"><span style="display: inline-block; position: relative; width: 1.806em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.52em, 4.178em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-4" style="font-family: MathJax_Math; font-style: italic;">x</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.572em;"><span class="texatom" id="MathJax-Span-5"><span class="mrow" id="MathJax-Span-6"><span class="mi" id="MathJax-Span-7" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span><span class="mo" id="MathJax-Span-8" style="font-size: 70.7%; font-family: MathJax_Main;">−</span><span class="mn" id="MathJax-Span-9" style="font-size: 70.7%; font-family: MathJax_Main;">1</span></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.312em; border-left: 0px solid; width: 0px; height: 0.905em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mrow class="MJX-TeXAtom-ORD"><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></math></span></span><script type="math/tex" id="MathJax-Element-1"> x _ {t-1} </script> ，<span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-2-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>x</mi><mrow class=&quot;MJX-TeXAtom-ORD&quot;><mi>t</mi></mrow></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-10" style="width: 1.096em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.885em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1000.89em, 2.502em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-11"><span class="msubsup" id="MathJax-Span-12"><span style="display: inline-block; position: relative; width: 0.902em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.52em, 4.178em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-13" style="font-family: MathJax_Math; font-style: italic;">x</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.572em;"><span class="texatom" id="MathJax-Span-14"><span class="mrow" id="MathJax-Span-15"><span class="mi" id="MathJax-Span-16" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.252em; border-left: 0px solid; width: 0px; height: 0.845em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mrow class="MJX-TeXAtom-ORD"><mi>t</mi></mrow></msub></math></span></span><script type="math/tex" id="MathJax-Element-2"> x _ {t} </script> ， <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-3-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>x</mi><mrow class=&quot;MJX-TeXAtom-ORD&quot;><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-17" style="width: 2.19em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.823em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1001.82em, 2.552em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-18"><span class="msubsup" id="MathJax-Span-19"><span style="display: inline-block; position: relative; width: 1.806em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.52em, 4.178em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-20" style="font-family: MathJax_Math; font-style: italic;">x</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.572em;"><span class="texatom" id="MathJax-Span-21"><span class="mrow" id="MathJax-Span-22"><span class="mi" id="MathJax-Span-23" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span><span class="mo" id="MathJax-Span-24" style="font-size: 70.7%; font-family: MathJax_Main;">+</span><span class="mn" id="MathJax-Span-25" style="font-size: 70.7%; font-family: MathJax_Main;">1</span></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.312em; border-left: 0px solid; width: 0px; height: 0.905em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mrow class="MJX-TeXAtom-ORD"><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub></math></span></span><script type="math/tex" id="MathJax-Element-3"> x _ {t+1} </script>就是不同时刻的输入，每个x都具有input layer的n维特征，依次进入循环神经网络以后，隐藏层输出<span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-4-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>s</mi><mi>t</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-26" style="width: 0.94em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.781em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1000.78em, 2.502em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-27"><span class="msubsup" id="MathJax-Span-28"><span style="display: inline-block; position: relative; width: 0.799em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.42em, 4.177em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-29" style="font-family: MathJax_Math; font-style: italic;">s</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.469em;"><span class="mi" id="MathJax-Span-30" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.252em; border-left: 0px solid; width: 0px; height: 0.845em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mi>t</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-4">s_t</script>受到上一时刻<span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-5-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>s</mi><mrow class=&quot;MJX-TeXAtom-ORD&quot;><mi>t</mi><mo>&amp;#x2212;</mo><mn>1</mn></mrow></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-31" style="width: 2.086em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.719em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1001.72em, 2.552em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-32"><span class="msubsup" id="MathJax-Span-33"><span style="display: inline-block; position: relative; width: 1.703em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.42em, 4.177em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-34" style="font-family: MathJax_Math; font-style: italic;">s</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.469em;"><span class="texatom" id="MathJax-Span-35"><span class="mrow" id="MathJax-Span-36"><span class="mi" id="MathJax-Span-37" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span><span class="mo" id="MathJax-Span-38" style="font-size: 70.7%; font-family: MathJax_Main;">−</span><span class="mn" id="MathJax-Span-39" style="font-size: 70.7%; font-family: MathJax_Main;">1</span></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.312em; border-left: 0px solid; width: 0px; height: 0.905em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></math></span></span><script type="math/tex" id="MathJax-Element-5">s_{t-1}</script>的隐藏层输出以及此刻输入层输入<span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-6-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>x</mi><mi>t</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-40" style="width: 1.096em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.885em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.589em, 1000.89em, 2.502em, -1000em); top: -2.188em; left: 0em;"><span class="mrow" id="MathJax-Span-41"><span class="msubsup" id="MathJax-Span-42"><span style="display: inline-block; position: relative; width: 0.902em; height: 0px;"><span style="position: absolute; clip: rect(3.412em, 1000.52em, 4.178em, -1000em); top: -4.01em; left: 0em;"><span class="mi" id="MathJax-Span-43" style="font-family: MathJax_Math; font-style: italic;">x</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.86em; left: 0.572em;"><span class="mi" id="MathJax-Span-44" style="font-size: 70.7%; font-family: MathJax_Math; font-style: italic;">t</span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.188em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.252em; border-left: 0px solid; width: 0px; height: 0.845em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mi>t</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-6">x_t</script> 的两方影响。 <br>
如果要更详细地了解tensorflow对RNN的解释，清戳官方<a href="https://www.tensorflow.org/tutorials/recurrent" target="_blank">tensorflow.RNN</a> <br>
另外推荐的学习资料：<a href="http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/" target="_blank">WildML</a></p>



<h2 id="什么是lstm"><a name="t2"></a>什么是LSTM</h2>

<p>LSTM全称长短期记忆人工神经网络（Long-Short Term Memory），是对RNN的变种。举个例子，假设我们试着去预测“I grew up in France… 中间隔了好多好多字……I speak fluent <strong>__</strong>”下划线的词。我们拍脑瓜子想这个词应该是French。对于循环神经网络来说，当前的信息建议下一个词可能是一种语言的名字，但是如果需要弄清楚是什么语言，我们是需要离当前下划线位置很远的“France” 这个词信息。相关信息和当前预测位置之间的间隔变得相当的大，在这个间隔不断增大时，RNN 会丧失学习到连接如此远的信息的能力。 <br>
这个时候就需要LSTM登场了。在LSTM中，我们可以控制丢弃什么信息，存放什么信息。 <br>
具体的理论这里就不多说了，推荐一篇博文<a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks</a>，里面有对LSTM详细的介绍，有网友作出的翻译请戳<a href="http://www.jianshu.com/p/9dc9f41f0b29" target="_blank">[译] 理解 LSTM 网络</a></p>



<h1 id="股票预测"><a name="t3"></a><strong>股票预测</strong></h1>

<p>在对理论有理解的基础上，我们使用LSTM对股票每日最高价进行预测。在本例中，仅使用一维特征。 <br>
数据格式如下： <br>
<img src="https://img-blog.csdn.net/20170219192524045?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXlsb3ZlMDQxNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" title=""></p>

<p>本例取每日最高价作为输入特征[x]，后一天的最高价最为标签[y] <br>
获取数据，请戳<a href="https://pan.baidu.com/share/init?shareid=442975122&amp;uk=75017666" target="_blank">stock_dataset.csv</a>,密码：md9l</p>



<h2 id="导入数据"><a name="t4"></a>导入数据：</h2>



<pre class="prettyprint"><code class="language-python hljs  has-numbering"><span class="hljs-keyword">import</span> pandas <span class="hljs-keyword">as</span> pd
<span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">import</span> matplotlib.pyplot <span class="hljs-keyword">as</span> plt
<span class="hljs-keyword">import</span> tensorflow
f=open(<span class="hljs-string">'stock_dataset.csv'</span>)  
df=pd.read_csv(f)     <span class="hljs-comment">#读入股票数据</span>
data=np.array(df[<span class="hljs-string">'最高价'</span>])   <span class="hljs-comment">#获取最高价序列</span>
data=data[::-<span class="hljs-number">1</span>]      <span class="hljs-comment">#反转，使数据按照日期先后顺序排列</span>
<span class="hljs-comment">#以折线图展示data</span>
plt.figure()
plt.plot(data)
plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  <span class="hljs-comment">#标准化</span>
normalize_data=normalize_data[:,np.newaxis]  <span class="hljs-comment">#增加维度</span>
<span class="hljs-comment">#———————————————————形成训练集—————————————————————</span>
<span class="hljs-comment">#设置常量</span>
time_step=<span class="hljs-number">20</span>      <span class="hljs-comment">#时间步</span>
rnn_unit=<span class="hljs-number">10</span>       <span class="hljs-comment">#hidden layer units</span>
batch_size=<span class="hljs-number">60</span>     <span class="hljs-comment">#每一批次训练多少个样例</span>
input_size=<span class="hljs-number">1</span>      <span class="hljs-comment">#输入层维度</span>
output_size=<span class="hljs-number">1</span>     <span class="hljs-comment">#输出层维度</span>
lr=<span class="hljs-number">0.0006</span>         <span class="hljs-comment">#学习率</span>
train_x,train_y=[],[]   <span class="hljs-comment">#训练集</span>
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(len(normalize_data)-time_step-<span class="hljs-number">1</span>):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+<span class="hljs-number">1</span>:i+time_step+<span class="hljs-number">1</span>]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) </code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li><li style="color: rgb(153, 153, 153);">19</li><li style="color: rgb(153, 153, 153);">20</li><li style="color: rgb(153, 153, 153);">21</li><li style="color: rgb(153, 153, 153);">22</li><li style="color: rgb(153, 153, 153);">23</li><li style="color: rgb(153, 153, 153);">24</li><li style="color: rgb(153, 153, 153);">25</li><li style="color: rgb(153, 153, 153);">26</li><li style="color: rgb(153, 153, 153);">27</li><li style="color: rgb(153, 153, 153);">28</li></ul></pre>

<p>出来的train_x就是像这个样子：</p>



<pre class="prettyprint"><code class="hljs lua has-numbering"><span class="hljs-string">[[[-1.59618],……中间还有18个……, [-1.56340]]</span>
  ……
 <span class="hljs-string">[[-1.59202] [-1.58244]]</span>]</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li></ul></pre>

<p>是一个shape为[-1,time_step,input__size]的矩阵</p>



<h2 id="定义神经网络变量"><a name="t5"></a>定义神经网络变量</h2>



<pre class="prettyprint"><code class="language-python hljs  has-numbering">X=tf.placeholder(tf.float32, [<span class="hljs-keyword">None</span>,time_step,input_size])    <span class="hljs-comment">#每批次输入网络的tensor</span>
Y=tf.placeholder(tf.float32, [<span class="hljs-keyword">None</span>,time_step,output_size]) <span class="hljs-comment">#每批次tensor对应的标签</span>

<span class="hljs-comment">#输入层、输出层权重、偏置</span>
weights={
         <span class="hljs-string">'in'</span>:tf.Variable(tf.random_normal([input_size,rnn_unit])),
         <span class="hljs-string">'out'</span>:tf.Variable(tf.random_normal([rnn_unit,<span class="hljs-number">1</span>]))
         }
biases={
        <span class="hljs-string">'in'</span>:tf.Variable(tf.constant(<span class="hljs-number">0.1</span>,shape=[rnn_unit,])),
        <span class="hljs-string">'out'</span>:tf.Variable(tf.constant(<span class="hljs-number">0.1</span>,shape=[<span class="hljs-number">1</span>,]))
        }</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li></ul></pre>



<h2 id="定义lstm网络"><a name="t6"></a>定义lstm网络</h2>



<pre class="prettyprint"><code class="hljs python has-numbering"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">lstm</span><span class="hljs-params">(batch)</span>:</span>  <span class="hljs-comment">#参数：输入网络批次数目</span>
    w_in=weights[<span class="hljs-string">'in'</span>]
    b_in=biases[<span class="hljs-string">'in'</span>]
    input=tf.reshape(X,[-<span class="hljs-number">1</span>,input_size])  <span class="hljs-comment">#需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入</span>
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-<span class="hljs-number">1</span>,time_step,rnn_unit])  <span class="hljs-comment">#将tensor转成3维，作为lstm cell的输入</span>
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  <span class="hljs-comment">#output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果</span>
    output=tf.reshape(output_rnn,[-<span class="hljs-number">1</span>,rnn_unit]) <span class="hljs-comment">#作为输出层的输入</span>
    w_out=weights[<span class="hljs-string">'out'</span>]
    b_out=biases[<span class="hljs-string">'out'</span>]
    pred=tf.matmul(output,w_out)+b_out
    <span class="hljs-keyword">return</span> pred,final_states</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li></ul></pre>



<h2 id="训练模型"><a name="t7"></a>训练模型</h2>



<pre class="prettyprint"><code class="hljs python has-numbering"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">train_lstm</span><span class="hljs-params">()</span>:</span>
    <span class="hljs-keyword">global</span> batch_size
    pred,_=rnn(batch_size)
    <span class="hljs-comment">#损失函数</span>
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-<span class="hljs-number">1</span>])-tf.reshape(Y, [-<span class="hljs-number">1</span>])))
 train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    <span class="hljs-keyword">with</span> tf.Session() <span class="hljs-keyword">as</span> sess:
        sess.run(tf.global_variables_initializer())
        <span class="hljs-comment">#重复训练10000次</span>
        <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">10000</span>):
            step=<span class="hljs-number">0</span>
            start=<span class="hljs-number">0</span>
            end=start+batch_size
            <span class="hljs-keyword">while</span>(end&lt;len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                <span class="hljs-comment">#每10步保存一次参数</span>
                <span class="hljs-keyword">if</span> step%<span class="hljs-number">10</span>==<span class="hljs-number">0</span>:
                    print(i,step,loss_)
                    print(<span class="hljs-string">"保存模型："</span>,saver.save(sess,<span class="hljs-string">'stock.model'</span>))
                step+=<span class="hljs-number">1</span></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li><li style="color: rgb(153, 153, 153);">19</li><li style="color: rgb(153, 153, 153);">20</li><li style="color: rgb(153, 153, 153);">21</li><li style="color: rgb(153, 153, 153);">22</li><li style="color: rgb(153, 153, 153);">23</li></ul></pre>



<h2 id="预测模型"><a name="t8"></a>预测模型</h2>



<pre class="prettyprint"><code class="hljs python has-numbering"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">prediction</span><span class="hljs-params">()</span>:</span>
    pred,_=lstm(<span class="hljs-number">1</span>)    <span class="hljs-comment">#预测时只输入[1,time_step,input_size]的测试数据</span>
    saver=tf.train.Saver(tf.global_variables())
    <span class="hljs-keyword">with</span> tf.Session() <span class="hljs-keyword">as</span> sess:
        <span class="hljs-comment">#参数恢复</span>
        module_file = tf.train.latest_checkpoint(base_path+<span class="hljs-string">'module2/'</span>)
        saver.restore(sess, module_file) 
        <span class="hljs-comment">#取训练集最后一行为测试样本。shape=[1,time_step,input_size]</span>
        prev_seq=train_x[-<span class="hljs-number">1</span>]
        predict=[]
        <span class="hljs-comment">#得到之后100个预测结果</span>
        <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">100</span>):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-<span class="hljs-number">1</span>])
            <span class="hljs-comment">#每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本</span>
            prev_seq=np.vstack((prev_seq[<span class="hljs-number">1</span>:],next_seq[-<span class="hljs-number">1</span>]))
        <span class="hljs-comment">#以折线图表示结果</span>
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color=<span class="hljs-string">'b'</span>)
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color=<span class="hljs-string">'r'</span>)
        plt.show()</code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li><li style="color: rgb(153, 153, 153);">5</li><li style="color: rgb(153, 153, 153);">6</li><li style="color: rgb(153, 153, 153);">7</li><li style="color: rgb(153, 153, 153);">8</li><li style="color: rgb(153, 153, 153);">9</li><li style="color: rgb(153, 153, 153);">10</li><li style="color: rgb(153, 153, 153);">11</li><li style="color: rgb(153, 153, 153);">12</li><li style="color: rgb(153, 153, 153);">13</li><li style="color: rgb(153, 153, 153);">14</li><li style="color: rgb(153, 153, 153);">15</li><li style="color: rgb(153, 153, 153);">16</li><li style="color: rgb(153, 153, 153);">17</li><li style="color: rgb(153, 153, 153);">18</li><li style="color: rgb(153, 153, 153);">19</li><li style="color: rgb(153, 153, 153);">20</li><li style="color: rgb(153, 153, 153);">21</li></ul></pre>



<h1 id="代码"><a name="t9"></a><strong>代码</strong></h1>

<p><a href="https://github.com/LouisScorpio/datamining/tree/master/tensorflow-program/rnn/stock_predict" target="_blank">完整代码</a></p>

<p>这一讲只有把最高价作为特征，去预测之后的最高价趋势，下一讲会增加输入的特征维度，把最低价、开盘价、收盘价、交易额等作为输入的特征对之后的最高价进行预测。</p>

<p>注：本文在介绍RNN和LSTM的部分，出处若涉及版权问题或原文链接错误，请指正，必会马上修改。</p>                