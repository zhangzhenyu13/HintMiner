<!DOCTYPE html>
<html>
<body>
  <div id="readme" class="readme blob instapaper_body">
    <article class="markdown-body entry-content" itemprop="text"><h1><a id="user-content-improving-ir-based-bug-localization-with-context-aware-query-reformulation" class="anchor" aria-hidden="true" href="#improving-ir-based-bug-localization-with-context-aware-query-reformulation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>HintMiner: Automatic Question Hints Mining Through Web Q\&A Resources Comprehension and Generation</h1>
<h2><a id="user-content-accepted-paper-at-esecfse-2018" class="anchor" aria-hidden="true" href="#accepted-paper-at-esecfse-2018"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>The Source Code For HintMiner Project</h2>

<pre><code>
The Project introduces a novel approach leveraging a light LLM to aquire exciting results in retrieval augmentation based generation paradigm via a basic semantic unit (i.e. sentence) level re-reconstruction SSL, which is quites different from previous works.
The work is accepted by AISTATS 2024 as regular long paper in main conference. AISTATS is one the four top tier Machine Learning conference (https://github.com/lixin4ever/Conference-Acceptance-Rate).

Cite us if the project is helpful to You.
</code></pre>
<pre><code>@article{HintMiner, 
author={Zhenyu Zhang, JiuDong Yang}, 
title={HintMiner: Automatic Question Hints Mining From Q\&A Web Posts with Language Model via Self-Supervised Learning},
year={2024},
booktitle={The 27th International Conference on Artificial Intelligence and Statistics},
url={https://github.com/zhangzhenyu13/HintMiner} 
}
</pre></code>

    
<h2>
<a id="user-content-subject-systems-6" class="anchor" aria-hidden="true" href="#subject-systems-6"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Instruction for building the Q&A system from source code and executing experiments
</h2>
<ul>
<li>Prepare system environment</li>
<li>Models</li>
<li>Data Pipeline</li>
<li>Build Models</li>
<li>Deploy System</li>
<li>Evaluation Results</li>
</ul>

<h2><a id="user-content-materials-included" class="anchor" aria-hidden="true" href="#materials-included"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Prepare system environment
</h2>
<p><strong>Minimum configuration of machines</strong></p>

<ul>
<li><code>RAM:</code> 512G</li>
<li><code>CPU:</code> 56 logic cores</li>
<li><code>Disk:</code> 1TB+</li>
<li>GPU: 4X Tesla V100(32G X 4)</li>

</ul>
<p><strong>Install python environment</strong></p>
<p>We develop the whole system using python, so we recommend you to install an anaconda virtual python3.6 environment at: https://www.anaconda.com/
</p>

<p><strong>Install MongoDB Database</strong></p>
<p>
Install MongoDB database into your computer with a linux system, and configure db ip and port according to the instruction of https://www.mongodb.com/.
To enable, fast retrieval of those data, install an Elastic Search Engine according to the instruction of 
https://www.elastic.co/.
</p>

<p><strong>Required python packages</strong></p>
<ul>
<li><code>machine learning:</code>scikit-learn, tensorflow, openNMT,texar,pytorch,networkx,sumeval,summy,TextBlob,bert-as-service</li>
<li><code>data preprocessing:</code> pymongo, numpy, pandas</li>
</ul>

<p><strong>Project Check</strong></p>
<ul>
<li>Prepare Data</li>
<li>The mentioned neural network models are in repo/programmingalpha/models. </li>
<li>The evaluation metiric tool APIs are in repo/programmingalpha/Utility/metrics.py. </li>
<li>Run the scripts in repo/test/db_test/ folder to prepare training data. </li>
<li>Run the scripts in repo/test/text_generation_test/ to build the model mentioned in HintMiner. </li>
</ul>


<h2><a id="user-content-available-operations" class="anchor" aria-hidden="true" href="#available-operations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Prepare Data
</h2>
<p>Download the data dump from archieve.org. 
Our training data from 4 online Q&A forums currently consists of Stack Overflow, Artificail Intelligence, Cross Validated and Data Science. 
</p>
<ul>
<li>
Build a MongoDB cluster and put all the data needed to the Database. Then deploy the elastic search engine on top of your database cluster.
</li>
<li>
Make the dirs listed in repo/programmingalpha/__init__.py.
</li>
</ul>

<h2><a id="user-content-required-parameters-for-the-operations" class="anchor" aria-hidden="true" href="#required-parameters-for-the-operations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Download the BERT Model
</h2>
<p><strong>
As the project is heavily based on several open released pretraining models, we at least need to prepare the BERT models according to
the instructions of https://github.com/google-research/bert (tensorflow version) and https://github.com/huggingface/pytorch-pretrained-BERT (pytorch version).
Store the pretrained model weight and auxiliary data of BERT model to the dirs BertBasePath or BertLargePath mentioned in ProgrammingAlpha/programmingalpha/__init__.py.
</strong></p>

<h2><a id="user-content-required-parameters-for-the-operations" class="anchor" aria-hidden="true" href="#required-parameters-for-the-operations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Prepare the training Data
</h2>


<p><strong>Training Data for HintMiner </strong></p>
  <ul>
  <li>
  Run repo/test/db_test/gen_corpus_seq2seq.py and push the generated corpus to mongodb cluster.
  </li>
  <li>
  Run repo/test/db_test/gen_samples.py with task parameter as 'seq2seq' to sample training and validating data.
  </li>
  <li>
  Leverage the code snippets in OpenNMT package and generate training data. Instructions can be found here http://opennmt.net/OpenNMT-py/options/preprocess.html.
  </li>
  </ul>



<p><strong>Build Local Knowledge Base</strong></p>
  <ul>
  <li>
  Run repo/test/db_test/buildQAIndexer.py firstly to gather all answers to each question.
  </li>
  <li>
  Push the knowledge units data to mongoDB cluster.
  </li>
  </ul>
  
<p><strong>After finished running all the above scripts, the system is ready for model training.</strong></p>

<h2><a id="user-content-q1-how-to-install-the-blizzard-tool" class="anchor" aria-hidden="true" href="#q1-how-to-install-the-blizzard-tool"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Train Neural Network Models
</h2>
<p><strong>Build Text Generation Models (e.g. MiningNet)</strong></p>
<ul>
<li>
Run repo/test/text_generation_test/build_copy_transformer.py to begin teacher forcing training of MiningNet.
</li>
<li>
Run repo/test/text_generation_test/build_rl_transformer.py to start training MiningNet using reinforcement learning.
</li>
<li>
To train a text generation model with other networks, a quick start can be followed in http://opennmt.net/OpenNMT-py/options/train.html.
</li>
<li>
Other optional networks for text generation is also available in https://github.com/asyml/texar.
</li>
</ul>



<h2><a id="user-content-query-file-format" class="anchor" aria-hidden="true" href="#query-file-format"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Evaluate the Model Performance
</h2>
<p><strong>
 Evaluate the HintMiner 
</strong></p>
<ul>
<li>
Sample 2000 solved questions via runining "ProgrammingAlpha/test/db_test/gen_samples.py --maxSize 2000 --task seq2seq" or unsolved questions via ProgrammingAlpha/test/db_test/unsolved_seq2seq.py. Or you can directly invoke the Google Custom Search Engine after including the 4 online forums mentioned before.
</li>
<li>
After finishing training the MiningNet and other text generation models, use ProgrammingAlpha/test/text_generation_test/run_inference.sh or repo/test/text_generation_test/transformerinference.py to generate answers to the sampled questions.
</li>
<li>
Run repo/test/utilities_test/computeScore.py true_answers.file generated_answers.file to get the evaluation BLEU/ROUGUE-2 score.
</li>
<li>
We also have conducted a simple user survey using online web here https://wj.qq.com/s2/3597786/b668/.
And the resuls are listed below.
</li>
</ul>

### User Survey
| Id      | 2   | 1      | 0   | -1      | -2   | mean      | std.dev.   |
| ---| ---|---| ---|---| ---|----|----|
| 1 | 17 | 8 | 1  | 0 | 1 | 1.481 | 0.768 |
| 2 | 3 | 11 | 9 | 1 | 3  | 0.37 | 1.196  |
| 3 | 0 | 7 | 10 | 5 | 5 | -0.296 | 1.097  |
| 4 | 1 | 5 | 7 | 10 | 4 | -0.407 | 1.13  |
| 5 | 24 | 3 | 0 | 0 | 0 | 1.888 | 0.098  |
| 6 | 13 | 11 | 1 | 1 | 1 | 1.259 | 0.932 |
| 7 | 13 | 10 | 4 | 0 | 0 | 1.333 | 0.518 |
| 8 | 6 | 8 | 7 | 4 | 2 | 0.444 | 1.432  |
| 9 | 19 | 6 | 1 | 0 | 1 | 1.555 | 0.765 |
| 10 | 3 | 14 | 7 | 2 | 1 | 0.592 | 0.834 |
| total | 99 | 83 | 47 | 23 | 18 | 0.822 | 0.877 |


<p>..........................................................</p>


<h2><a id="user-content-please-cite-our-work-as" class="anchor" aria-hidden="true" href="#please-cite-our-work-as"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>
Deploying HintMiner
</h2>

<p><strong>User Interface</strong></p>
<ul>
<li>
We currently implemented a very simple answering outlook asking box, which is available in https://github.com/AnonymousAuthor2013/HintMiner/tree/master/alphaservices.
</li>
<li>
The restful API for text-generation network can be started following instrcutions here http://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392. 
</li>
<li>
The full one-shot deployment is under working now...
</li>
</ul>

  </body>
</html>

