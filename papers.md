# papers digested

wanted to gather and share my thoughts from reading various neural net papers

* [Weight Agnostic Neural Networks](https://arxiv.org/pdf/1906.04358.pdf)
  * interesting: 8/10
  * intuitive: 8/10
  * The traditional view of optimization in deep learning (and often in general) is that we are searching the space of weights to find the best ones. In other words, learning is a search problem. From the perspective of search, this paper is suggesting we search for the right architecture rather than the right set of weights. Instead of using gradient descent or some other method to find the right weights, we are leveraging our knowledge of existing architectures to find the right combination of architectures to properly solve a task. In addition, it's searching for minimal architectures and so methods such as network pruning could be leveraged. In the paper, the authors start from nothing and add architectures rather than prune from a fully trained network. Regardless, the idea is that too big of an architecture implies potential overfitting just as too big of a weight implies potential overfitting.
  * One of the key aspects of deep learning is that given a parameterized function, we can find weights to represent any function if it has sufficient depth and complexity. This paper suggests that the representational power of architectures is really strong, and suggests as strong in some cases as neural networks with optimizable weights. Basically determining the weights shouldn't matter as the paper title suggests. I do believe the two have equal representational power because they're really trying to model at different levels of granularity. (see below for in depth analysis and personal takeaway)
  * My personal takeaway is that we should consider taking different levels of granularity in our approach to modeling. Rather than modeling at the individual neuron level and tuning the weights, we are modeling at the collective neuron level. More specifically an example would be the Convolutional Layer or the LSTM Layer where you have multiple neurons with randomized weights/kernels. I would maybe compare this with thinking at the atomic level vs thinking at the chemical level (where chemicals are comprised of atoms but their own things in their own rights - eg. has reactive properties, etc.). Disclaimer: not a chemist
  * Another takeaway is that there are many ways to model something, and really it's about what we keep as fixed and what becomes the hyperparameter.

* [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf)
  * interesting: 7/10
  * intuitive: 7/10
  * helps understand this idea of generating an ideal image to visualize a class (optimizing on the image given a trained model with determined weights)
  * helps understand this idea of creating a "saliency map" which basically says which pixels are important for a given image, based on what one backprop pass tells us
* [STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf)
  * interesting: 6/10
  * intuitive: 7/10
  * we can replace max pooling layers with conv layers that have stride greater than 1, with similar state of the art results
  * we can use a new method of visualizing representations learned by the higher/deeper layers of a conv net that combines backprop and deconvnet approaches - dubbed "guided backprop" (naming could be better?)
    * idea is that deconvnet only focuses on positive impact to the output ie. positive gradient values
    * backprop is based on the activation function so for example for a relu, only the gradient values with the original input > 0 are passed back
    * combining these two, we pass back gradients that satisfy both ie. the original input needs to be > 0 (for a relu layer) _and_ the gradient value itself needs to be positive
    * we do this all the way to the original image input, and get the gradient with respect to the image pixels (and thus the "visual representation" of the importance of each pixel with respect to the particular layer output that we backprop'd from)
* [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
  * interesting: 6/10
  * intuitive: 6/10
  * idea here is that guided backprop applies to every backprop all the way to the input pixels whereas grad-CAM passes normal backprop until the last convolution layer, at which point you get the importance of each feature map activation (the gradient with respect to each feature map activation)(and apparently these are global average pooled across the entire feature map), weight each feature map activation by their respective feature map activation importance, and sum these up and shove these through a ReLU - this results in a coarse heatmap of the feature activation maps for the last convolutional layer - note: the result of grad-CAM is first upsampled to the input image resolution using bilinear interpolation before being combined with guided backprop
  * a feature map activation is just the activated feature map resulting from a convolution with k filters, and so there are k feature map activations
  * we combine grad-CAM with guided backprop to isolate the pixels that are prominent for the class in question, as opposed to just the pixels that we look at which are important (ie. salient maps and guided backprop)(they call these "high resolution", something that grad-CAM is not)
  * grad-CAM basically adds class based focus to these visualizations (ie. focus on class specific pixels)(i think the term they use is class-discriminative)
  * combined result is high resolution _and_ class-discriminative :tada:

* [Image Style Transfer using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
  * interesting: 8/10
  * intuitive: 7/10

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
  * interesting: 8.5/10 - i love the idea of training two separate neural networks, the encoder and the decoder for tasks like language translation. one of the main ideas is reducing something (like a language) into some fundamental representation (the job of the Encoder) and then to take that fundamental representation and decode it into a different language (the job of the Decoder)
  * intuitive: 9/10 - if you know RNNs/LSTMs, then this paper should be relatively straightforward
  * idea here is to train an Encoder to convert a sequence to its meaning/representation (ie. text to the meaning of the text in a vector) and then train a Decoder to convert an initial token plus the meaning of the text to a new sequence (trained obviously to look like the right new sequence).
  * an example would be language to language translation, where you train the Encoder to understand and capture the meaning in English and you train the Decoder to understand the meaning vector and convert that to a reasonable output sequence in the target language.
  * in the paper, we stack LSTMs because empirically it was shown that deep networks outperformed shallow networks
  * for this particular architecture, the number of layers in both the Encoder and Decoder must match in order to properly use all the context vectors - if not, we would need to average the context vectors or something in order to fit X context vectors from the Encoder into Y hidden states for the Decoder.
* [illustrated seq2sec with attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [illustrated transformer with attention](https://jalammar.github.io/illustrated-transformer/)
* [the annotated transformer: harvard nlp group dissecting the attn is all you need paper](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [actual paper for attn is all you need](https://arxiv.org/pdf/1706.03762.pdf)
  * interesting: 8.5/10 - learning to focus on the right part of a sentence is game changing, and in theory can be applied to not just transformers but to CNNs, GNNs, and other kinds of neural networks
  * intuitive: 7.5/10 - dense with lots of formulas, ideally you start with the annotated version of this paper first by the Harvard NLP group

* [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
  * interesting: 9.5/10 - it's mind boggling what the paper shows GPT-3 can do
  * intuitive: 6/10 - the paper is long and dense, if you have previous knowledge of seq2seq models and transformers, great, if not, well this paper might seem outlandish
  * idea here is that OpenAI trained a huge neural net to basically memorize the relationships between words learned across a huge collection of text scraped from the internet and this neural net model can be used in a variety of natural language processing tasks without much custom training
