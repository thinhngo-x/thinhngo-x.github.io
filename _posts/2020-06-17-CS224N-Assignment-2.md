---
layout: article
mathjax: true
title: "CS224n Assignment 2"
author: "Thinh Ngo"
tags: [CS224N, NLP, Machine Learning, Word2vec, Negative Sampling]
image: word_vectors.png
---

# 1. Written: Understanding word2vec

**Reminder**: In word2vec, a conditional probability of a word $o$ given a center word $c$ is given by:

$$
P(O=o|C=c) = \frac{\exp(u_o^\intercal v_c)}{\sum_{w \in Vocab}\exp(u_w^\intercal v_c)}
$$

The objective function of the optimization problem is deduced from the log-likelihood function:

$$
J_{\text{naive-softmax}}(v_c, o, U) = -\log P(O=o|C=c)
$$


1. Prove that this loss function is also viewed as the cross-entropy loss function. Indeed, $y$ is a one-hot vector with a $1$ for the true outside word $o$ and $0$ for everywhere else. Thus:
   
	$$
	-\sum_{w \in Vocab}y_w \log \hat{y}_w = -\log \hat{y}_o \quad \square
	$$

2. Compute the partial derivative of $J_\text{naive-softmax}(v_c, o, U)$ w.r.t $v_c$.

	$$
	\begin{align}\frac{\partial J}{\partial v_c} &= \frac{\partial}{\partial v_c}(-u_o^\intercal v_c+\log(\sum_{w \in Vocab}\exp(u_w^\intercal v_c))) \\&=-u_o+\frac{1}{\sum_{w \in Vocab}\exp(u_w^\intercal v_c)}\sum_{w \in Vocab}\frac{\partial \exp(u_w^\intercal v_c)}{\partial v_c} \\&=-u_o+\frac{\sum \exp(u_w^\intercal v_c) u_w}{\sum \exp(u_w^\intercal v_c)}\\&=-u_o+\sum_{w \in Vocab}\hat{y}_w u_w \\&=\sum_{w \in Vocab}u_w(\hat{y}_w-y_w)\\&=U(\hat{y}-y)\end{align}
	$$

3. Compute the partial derivatives of $J_\text{naive-softmax}(v_c, o, U)$ w.r.t $u_w$.

    Considering 2 cases:

	- $w$ is $o$:
    
		$$
		\begin{align}
		\frac{\partial J}{\partial u_o} &= -v_c + \frac{\exp(u_o^\intercal v_c)}{\sum \exp(u_w^\intercal v_c)} v_c \\
		&= v_c(\hat{y}_o - y_o)
		\end{align}
		$$
   - $w = \omega$ different to $o$:

		$$
		\begin{align}
		\frac{\partial J}{\partial u_\omega} &= \frac{\exp(u_\omega^\intercal v_c)}{\sum \exp(u_w^\intercal v_c)} v_c \\
		&= v_c(\hat{y}_\omega - y_\omega)
		\end{align}
		$$

   In conclusion, $\frac{\partial J}{\partial U} = v_c(\hat{y}-y)^\intercal$

4. Derivative of sigmoid function: $\frac{d \sigma(x)}{dx} = \sigma(x)(1-\sigma(x))$

5. Considering Negative Sampling loss:
	
	$$
	J_{neg-sample}(v_c, o, U) = -\log (\sigma(u_o^\intercal v_c)) - \sum_{k=1}^K \log (\sigma(-u_k^\intercal v_c))
	$$
	Compute the derivatives fo $J_{neg-sample}$ w.r.t $v_c$, $U$
	
	$$
	\begin{align}
	\frac{\partial J}{\partial v_c} &= - (1-\sigma(u_o^\intercal v_c))u_o + \sum_{k=1}^K (1-\sigma(-u_k^\intercal v_c))u_k \\
	\frac{\partial J}{\partial u_o} &= - (1-\sigma(u_o^\intercal v_c))v_c \\
	\frac{\partial J}{\partial u_k} &= (1-\sigma(-u_k^\intercal v_c))v_c
	\end{align}
	$$

	One advantage of negative-sampling compared to the naive-softmax is the computing complexity since we only to do the calculation on $k$ sample words instead of the whole vocabulary.



# 2. Implementing word2vec

Full code and results could be found [here](https://github.com/thinhngo-x/cs224n/tree/master/a2)

![Word2vec visualization](https://raw.githubusercontent.com/thinhngo-x/cs224n/master/a2/word_vectors.png)

