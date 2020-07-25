---
title: 自然言語処理
author: 浅川伸一
---

# 自然言語処理前史

1. 第一次ブーム 1960 年代
極度の楽観論: 辞書を丸写しすれば翻訳は可能だと思っていた，らしい...
2. 第二次ブーム 統計的自然言語処理
    -  [統計的言語モデル statistical language model](https://en.wikipedia.org/wiki/Language_model)
    - <a target="_blank" href="https://nlp.stanford.edu/manning/">Chris Manning (スタンフォード大学)</a>) and Schutze (1999) 著。定番の教科書 <a target="_blank" href="https://nlp.stanford.edu/fsnlp/">Fundations of Statistical Natural Language Processing</a>, あるいは
    <a target="_blank" href="https://nlp.stanford.edu/fsnlp/promo/">こちら</a>
    - もう一つ定評の教科書 <a target="_blank" href="https://web.stanford.edu/~jurafsky/">Jurafsky</a> 著) と Martin 著 <a target="_blank" href="https://web.stanford.edu/~jurafsky/slp3/">Speech and Language Processing</a> は <a target="_blank" href="https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf">改訂版</a> が出版されました。ニューラルネットワークによる言語モデルも載っています。

# 用語解説
- <a target="_blank" href="https://en.wikipedia.org/wiki/Bag-of-words_model">BoW</a>: Bag of Words 単語の袋。ある文章を表現する場合に，各単語の表現を集めて袋詰めしたとの意味。従って語順が考慮されません。"犬が男を噛んだ" と "男が犬を噛んだ" では同じ表現になります。LSA, LDA, fastText なども同じような表現を与えます。
- TF-IDF: 単語頻度 (Term Frequency) と 逆(Inverse) 文書頻度 (Document Frequency) で文書のベクトル表現を定義する手法です。何度も出現する単語は重要なので単語頻度が高い文書には意味があります。一方，全ての文書に出現する単語は重要とは言えないので単語の出現る文書の個数の逆数の対数変換を用います。このようにしてできた文章表現を TF-IDF と言います。

#  言語モデル Language model
- 文献では言語モデルを **LM** と表記される。
- [統計的言語モデル statistical language model](https://en.wikipedia.org/wiki/Language_model)。言語系列に確率を与えるモデルの総称。良い言語モデル LM は，有意味文に高い確率を与え，曖昧な文には低い確率を与える。言語モデルは人工知能の問題。
1. n-gram 言語モデル
2. 指標: BELU, perplexity
3. 課題: NER, POS, COL, Summary, QA, Translation


<!--
## 関連分野
系列情報処理モデルには各分野で多くの試みがなされている。たとえば

1. 状態空間モデル (SSM), 隠れマルコフモデル (Hidden Markov models: HMM)
2. 自己回帰モデル (AR, ARMA, ARIMA, Box=Jenkins)
3. フィルタリング理論: カルマンフィルタ (Kalman filters), 粒子フィルタ(経済学部矢野浩一先生による[粒子フィルタの解説論文](https://www.terrapub.co.jp/journals/jjssj/pdf/4401/44010189.pdf))
3. ニューラルネットワーク
-->

# n-グラム言語モデル

- 類似した言語履歴 $h$ について, n-gram 言語モデルは言語履歴 $h$ によって言語が定まることを言います。
- 実用的には n-gram 言語モデルは $n$ 語の単語系列パターンを表象するモデルです。
- n-gram 言語モデルでは $n$ の次数増大に従って，パラメータは指数関数的に増大します。
- すなわち高次 n グラム言語モデルのパラメータ推定に必要な言語情報のコーパスサイズは，次数増大に伴って，急激不足します
- Wikipedia からの引用では次式:
$$
p(w_1,\dots,w_m)=\prod_{i=1}^{m} P(w_i\vert w_1,\ldots,w_{i-1})\simeq \prod_{i=1}^{m}p(w_i\vert w_{i-(n-1)},\ldots,w_{i-1})
$$
- 上式では $m$ ですが，伝統的に $n$ グラムと呼びます。$n=1$ であれば直前の 1 つを考慮して
次語を予測することになります。

<!--
- n-グラム言語モデル: 文脈 $h$ の中で単語 $w$ が何回出現したかをカウント。観測した全ての文脈 $h$ で正化
- 伝統的解: n-グラム言語モデル: $P\left(w\vert h\right)=\displaystyle\frac{C\left(h,w\right)}{C\left(h\right)}$
- 確率 $p\left(w_n\vert w_{1},\ldots,w_{n-1}\right)$
-->

<!-- # from Manning (1999) page 191.

In such a stochastic problem, we use a classification of the previous
words, the _history_ to predict the next word. On the basis of having looked
at a lot of text, we know which words tend to follow other words.

For this task, we cannot possibly consider each textual history separately:
most of the time we will be listening to a sentence that we have
never heard before, and so there is no previous identical textual history
on which to base our predictions, and even if we had heard the beginning
of the sentence before, it might end differently this time. And so we
-->

余談[^gram] ですが

- $n=0$: ヌルグラム null-gram 
- $n=1$: ユニグラム uni-gram
- $n=2$: バイグラム bi-gram
- $n=3$: トリグラム tri-gram

などと呼ばれます。

[^gram]: 五月蝿いことを言えば Manning (1999, p.193) によると単語 _gram_ はギリシャ語由来の単語だそうです。従って _gram_ に付ける数接頭辞もギリシャ語である教養を持つべきです。そうすると $n=1$: mono-gram, $n=2$: di-gram, $n=4$: tetra-gram が教養です。$n=3$ はギリシャ，ローマ共通で tri-gram です。日常会話では $n=4$ をクワッドグラム(ラテン語由来)やフォーグラムと呼ぶことも多いです。

<!--
The cases of n-gram models that people usually use are for $n=2,3,4$ and
these alternatives are usually referred to as a bigram, a trigram
four-gram, model, respectively. Revealing this will surely be enough to
cause any Classicists who are reading this book to stop, and to leave the
field to uneducated engineering sorts: is a _gram_ is a Greek root and so
should be put together with Greek number prefixes.  Shannon actually did
use the term but with dtigram, with declining levels of education in recent
decades, this usage has not survived. As non-prescriptive linguists,
however, we think that the curious mixture of English, Greek, and Latin
that our collegues actuall use is quite fun.  So we will try to stamp it out.
Rather than _four-gram_, some people do make an attempt at appearing educated
by saying _quad-gram_, but this is not really correct use of a Latin number
prefix (which would give _quadgram_ cf. _quadilateral_), let alone correct
use of a Greek number prefix, which would give us "a _tetragram_ model.”
that we have to specify to determine a particular model within that model space.
-->


## ニューラルネットワーク言語モデルあるいはミコロフ革命
<center>
<img src="/assets/Mikolov_portrait.jpg" style="width:24%">
<img src="/assets/2015Mikolov_NIPSportrait.png" style="width:33%"><br>
</center>

- スパースな言語履歴 $h$ は低次元空間へと射影される。類似した言語履歴は群化する
- 類似の言語履歴を共有することで，ニューラルネットワーク言語モデルは頑健
(訓練データから推定すべきパラメータが少ない)。

## ニューラルネットワーク言語モデル NNLM フィードフォワード型 NNLM

<center>
<img src="/assets/2012Mikolov_Google_Slides8.svg" style="width:74%"></br>
図: フィードフォワード型ニューラルネットワーク言語モデル NNLM [@2003Bengio],[@2007Schwenk].
</center>

## リカレントニューラルネットワーク言語モデル RNNLM

<center>
<img src="/assets/2011Mikolov_Extention_Fig1.svg" style="width:74%"></br>
</center>

- 入力層 $w$ と出力層 $y$ は同一次元，総語彙数に一致。(約一万語から20万語)
- 中間層 $s$ は相対的に低次元 (50から1000ニューロン)
- 入力層から中間層への結合係数行列 $U$，中間層から出力層への結合係数行列 $V$，
- 再帰結合係数行列 $W$ がなければバイグラム(2-グラム)ニューラルネットワーク言語モデルと等しい

<!--
## Model description - recurrent NNLM

<center>
<img src="../assets/2011Mikolov_Extention_Fig1.svg" style="width:74%">
</center>

- Input layer $w$ and output layer $y$ have the same dimensionality as the vocabulary (10K - 200K)
- Hidden layer $s$ is orders of magnitude smaller (50 - 1000 neurons)
- $U$ is the matrix of weights between input and hidden layer, $V$ is the matrix of weights between hidden and output layer
- Without the recurrent weights $W$, this model would be a bigram neural network language model

### ニューラルネットワーク言語モデル(4) リカレントニューラルネットワーク言語モデル RNNLM(2)

- 中間層ニューロンの出力と出力層ニューロンの出力は，それぞれ以下のとおり：

$$
s(t) = f\left(\mathbf{Uw}(t) + \mathbf{W}\right)s\left(t-1\right)\\
y(t) = g\left(\mathbf{Vs}\left(t\right)\right),
$$

$f(z)$ はシグモイド関数，$g(z)$ はソフトマックス関数。
P
最近のほとんどのニューラルネットワークと同じく出力層にはソフトマックス関数を用いる。
出力を確率分布とみなすように，全ニューロンの出力確率を合わせると1となるように

$$
f(z)=\frac{1}{1+e^{-z}}, 
$$

$$
g\left(z_m\right)=\frac{e^{z_{m}}}{\sum_k e^{z_{k}}}
$$

## Model Description - Recurrent NNLM
The output values from neurons in the hidden and output layers
are computed as follows:
$$
s(t) = f\left(\mathbf{Uw}(t) + \mathbf{W}\right)s\left(t-1\right)
$$

$$
y(t) = g\left(\mathbf{Vs}\left(t\right)\right),
$$
where $f\left(z\right)$ and $g\left(z\right)$ are sigmoid and softmax activation
unctions (the softmax function in the output layer is used to
ensure that the outputs form a valid probability distribution, i.e.
all outputs are greater than 0 and their sum is 1):
$$
f\left(z\right) =\frac{1}{1+e^{-z}}, g\left(z_m\right)=\frac{e^{z_{m}}}{\sum_ke^{z_{k}}}
$$

## RNNLM の学習(1)
- 確率的勾配降下法 (SGD)
- 全訓練データを繰り返し学習，結合係数行列 $U$, $V$, $W$ をオンライン学習 (各単語ごとに逐次)
- 数エポック実施 (通常 5-10)

### Training of RNNLM
- The training is performed using Stochastic Gradient Descent (SGD)
- We go through all the training data iteratively, and update the weight
- matrices $U$, $V$ and $W$ online (after processing every word)
- Training is performed in several epochs (usually 5-10)

### RNNLM の学習(2)

時刻 $t$ における出力層の誤差ベクトル $\mathbf{e}_o\left(t\right)$ の勾配計算には

クロスエントロピー誤差を用いて：

$$
\mathbf{e}_o\left(t\right) = \mathbf{d}\left(t\right)-\mathbf{y}\left(t\right)
$$

$\mathbf{d}(t)$ は出力単語を表すターゲット単語であり
時刻 $t+1$ の入力単語 $\mathbf{w}\left(t+1\right)$ (ビショップは PRML~\citep{PRML}では
1-of-ｋ 表現と呼んだ。ベンジオはワンホットベクトルと呼ぶ)。

### Training of RNNLM
radient of the error vector in the output layer $\mathbf{e}_o\left(t\right)$ is
computed using a cross entropy criterion:
$$
\mathbf{e}_o\left(t\right) = \mathbf{d}\left(t\right)-\mathbf{y}\left(t\right)
$$
where $\mathbf{d}\left(t\right)$ is a target vector that represents the word
$\mathbf{w}\left(t+1\right)$ (encoded as 1-of-V vector(ワンホットベクター)).

### RNNLM の学習(3)

時刻 $t$ における中間層から出力層への結合係数行列 $V$ は，
中間層ベクトル $\mathbf{s}\left(t\right) と出力層ベクトル $\mathbf{y}\left(t\right)$ を用いて
次式のように計算する

$$
\mathbf{V}(t+1) = \mathbf{V}(t) + \alpha\mathbf{s}(t)\mathbf{e}_o(t)^{\top}
$$

$\alpha$ は学習係数

# Training of RNNLM
Weights $V$ between the hidden layer $\mathbf{s}\left(t\right)$ and the output layer
$\mathbf{y}\left(t\right)$ are updated as

$$
\mathbf{V}\left(t+1\right)=\mathbf{V}\left(t\right)+\alpha\mathbf{s}\left(t\right)\mathbf{e}_o\left(t\right)^{\top}
$$
where $\alpha$ is the learning rate.

# RNNLM の学習(4)

続いて，出力層からの誤差勾配ベクトルから中間層の誤差勾配ベクトルを計算すると，

$$
\mathbf{e}_h\left(t\right) = d_{h}\left(\mathbf{e}_o\left(t\right)^{\top}\mathbf{V},t\right),
$$

誤差ベクトルは関数 $d_h()$ をベクトルの各要素に対して適用して

$$
d_{hj}\left(x,t\right) = x s_j\left(t\right)\left(1-s_{j}\left(t\right)\right)
$$

Next, gradients of errors are propagated from the output layer to the hidden layer

$$
\mathbf{e}_h\left(t\right) = d_{h}\left(\mathbf{e}_o\left(t\right)^{\top}\mathbf{V},t\right),
$$

where the error vector is obtained using function $d_h()$ that is applied element-wise

$$
d_{hj}\left(x,t\right) = x s_j\left(t\right)\left(1-s_{j}\left(t\right)\right)
$$

### RNNLM の学習(5)

時刻 $t$ における入力層から中間層への結合係数行列 $\mathbf{U}$ は，ベクトル $\mathbf{s}\left(t\right)$ の更新を以下のようにする。

$$
\mathbf{U}(t+1) = \mathbf{U}(t) + \alpha\mathbf{w}(t)\mathbf{e}_{h}(t)^{\top}
$$

時刻 $t$ における入力層ベクトル $\mathbf{w}(t)$ は，一つのニューロンを除き全て $0$ である。
上式のように結合係数を更新するニューロンは入力単語に対応する
一つのニューロンのそれを除いて全て $0$ なので，計算は高速化できる。

Weights $\mathbf{U}$ between the input layer $\mathbf{w}\left(t\right)$ and
the hidden layer $\mathbf{s}\left(t\right)$ are then updated as

$$
\mathbf{U}\left(t+1\right) = \mathbf{U}\left(t\right) + \alpha\mathbf{w}\left(t\right)\mathbf{e}_{h}\left(t\right)^{\top}
$$

Note that only one neuron is active at a given time in the input vector
$\mathbf{w}\left(t\right)$. As can be seen from the equation (\ref{eq:8}),
the weight change for neurons with zero activation is none, thus the
computation can be speeded up by updating weights that correspond just to
the active input neuron.
-->

## RNNLM の学習 時間貫通バックプロパゲーション BPTT

<center>
<img src="/assets/2011Mikolov_Extention_Fig3.svg" style="width:94%"><br>
2011 Mikolov Fig.3
</center>

- 再帰結合係数行列 $\mathbf{W}$ を時間展開し，多層ニューラルネットワークとみなして学習を行う
- 時間貫通バックプロパゲーションは Backpropagation Through Time (BPTT) といいます

<!--
### RNNLM の学習(8) 時間貫通バックプロパゲーション BPTT(3)

誤差伝播は再帰的に計算する。
通時バックプロパゲーションの計算方法では，
前の時刻の中間層の状態を保持しておく必要がある。

$$
{e}_h\left(t-\tau-1\right)d_{h}\left(e_h\left(t-\tau\right)^{\top}\mathbf{W},t-\tau-\right),
$$

時間展開したこの図で示すように各タイムステップで，繰り返し（再帰的に）で微分して
勾配ベクトルの計算が行われる。このとき各タイムステップの時々刻々の刻みを経るごとに
急速に勾配が小さくなってしまう
**勾配消失** が起きる。
活性化関数がロジスティック関数

$f(x)=\sigma(x)=\left(1+\exp(-x)\right)$

であれば，その微分は

$\sigma'(x)=x\left(1-x\right)$

であった。

ハイパータンジェント
$f(x)=\phi(x)=\left(e^{x}-e^{-x}\right)/\left(e^{x}+e^{-x}\right)$ であれば

$\phi'\left(x\right)=x\left(1-x^2\right)$ であるから，いずれの活性化関数を用いる場合でも
ニューロン$x$の値域（取りうる値）が $x=\left(x\vert 0\le x\le1\right)$
である限り，ロジスティック関数であれハイパータンジェント関数であれ，
元の値より $0$ に近い値となる。

本日は省略するが，これと反対の現象**勾配爆発** が起きる可能性がある。

**BPTT** で時刻に関する再帰が深いと深刻な問題となり
収束しない，学習がいつまで経っても終わらないことがある。

The unfolding can be applied for as many time steps as many training examples were already seen, however the error gradients quickly {\bf vanish} as they get backpropagated in time (in rare cases the errors can {\bf explode}), so several steps of unfolding are sufficient (this is sometimes referred to as __truncated BPTT__.

## RNNLM の学習(9) 時間貫通バックプロパゲーション BPTT(4)
再帰結合係数行列 $\mathbf{W}$ の更新には次の式を用いる：

$$
\mathbf{W}\left(t+1\right) = \mathbf{W}\left(t\right) + \alpha\sum_{z=0}^{T}\mathbf{s}\left(t-z-1\right)\mathbf{e}_{h}\left(t-z\right)^{\top}.
$$

行列 $\mathbf{W}$ の更新は誤差が逆伝播するたびに更新されるのではなく，一度だけ更
新する。そうしないと，どの時間を遡及している最中にどの時刻で計算した誤差によっ
て再帰結合行列を更新するのかという，更新の順番が影響する **タイムマシン問題** が発生する。
未来の子孫が過去の祖先を殺すと子孫は存在しえない。

計算効率の面からも，訓練事例をまとめて扱い，時間ステップニューラルネットワー
クの時刻 $T$ に関する時間展開に関する複雑さは抑えることが行われる。

The recurrent weights $\mathbf{W}$ are updated as

$$
\mathbf{W}\left(t+1\right) = \mathbf{W}\left(t\right) + \alpha\sum_{z=0}^{T}\mathbf{s}\left(t-z-1\right)\mathbf{e}_{h}\left(t-z\right)^{\top}.
$$

Note that the matrix $\mathbf{W}$ is changed in one update at once, and not
during backpropagation of errors.

%% It is more computationally efficient to unfold the network after processing
%% several training examples, so that the training complexity does not
%% increase linearly with the number of time steps $T$ for which the network
%% is unfolded in time.

-->

## 時間貫通バックプロパゲーション BPTT

<center>
<img src="/assets/2012Mikolov_Google_Slides20.svg" style="width:74%"><br>
図: バッチ更新の例。赤い矢印は誤差勾配がリカレントニューラルネットワーク
</center>
の時間展開を遡っていく様子を示している。

## 文字ベースか単語ベースか？
1. Pros/Cons
1. OOV problems。OOV: Out of Vocabulary 問題。ソーシャルメディアなどを活用する場合不可避の問題

