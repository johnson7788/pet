# Pattern-Exploiting Training (PET)
该repository包含代码for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676) and [It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118)
论文介绍了pattern-exploiting training（PET），这是一种半监督训练程序，
将输入示例重新编写为填空样式短语。在低资源环境中，尽管参数比GPT-3少99.9％，但PET和iPET明显优于常规监督训练，各种半监督基线甚至GPT-3。 
PET的迭代变体（iPET）可以训练多代模型，甚至可以在没有任何训练数据的情况下使用。

<table>
    <tr>
        <th>#Examples</th>
        <th>Training Mode</th>
        <th>Yelp (Full)</th>
        <th>AG's News</th>
        <th>Yahoo Questions</th>
        <th>MNLI</th>
    </tr>
    <tr>
        <td rowspan="2" align="center"><b>0</b></td>
        <td>unsupervised</td>
        <td align="right">33.8</td>
        <td align="right">69.5</td>
        <td align="right">44.0</td>
        <td align="right">39.1</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>56.7</b></td>
        <td align="right"><b>87.5</b></td>
        <td align="right"><b>70.7</b></td>
        <td align="right"><b>53.6</b></td>
    </tr>
    <tr>
        <td rowspan="3" align="center"><b>100</b></td>
        <td>supervised</td>
        <td align="right">53.0</td>
        <td align="right">86.0</td>
        <td align="right">62.9</td>
        <td align="right">47.9</td>
    </tr>
    <tr>
        <td>PET</td>
        <td align="right">61.9</td>
        <td align="right">88.3</td>
        <td align="right">69.2</td>
        <td align="right">74.7</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>62.9</b></td>
        <td align="right"><b>89.6</b></td>
        <td align="right"><b>71.2</b></td>
        <td align="right"><b>78.4</b></td>
    </tr>
</table>
    
<sup>*Note*: To exactly reproduce the above results, make sure to use v1.1.0 (`--branch v1.1.0`).</sup>

## 📑 Contents

**[🔧 Setup](#-setup)**

**[💬 CLI Usage](#-cli-usage)**

**[💻 API Usage](#-api-usage)**

**[🐶 Train your own PET](#-train-your-own-pet)**

**[📕 Citation](#-citation)**

## 🔧 Setup

有关PET的所有requirements都可以在“requirements.txt”中找到。您可以使用`pip install -r requirements.txt`安装所有必需的软件包。

## 💬 CLI Usage

该存储库中的命令行界面cli.py当前支持三种不同的训练pattern（PET，iPET，有监督的训练），两种其他评估方法（无监督和priming）以及13种不同的任务。
有关Yelp Reviews，AG's News，Yahoo Questions，MNLI和X-Stance，请参阅https://arxiv.org/abs/2001.07676 以获取更多详细信息。
有关8个SuperGLUE任务，请参见https://arxiv.org/abs/2009.07118

### PET Training and Evaluation
训练和评估PET模型, 需运行以下命令：

    python3 cli.py \
	--method pet \
	--pattern_ids $PATTERN_IDS \
	--data_dir $DATA_DIR \
	--model_type $MODEL_TYPE \
	--model_name_or_path $MODEL_NAME_OR_PATH \
	--task_name $TASK \
	--output_dir $OUTPUT_DIR \
	--do_train \
	--do_eval
    
 其中 
 - `$PATTERN_IDS` 指定要使用的PVP。例如，如果要使用 *all* pattern，则为AG's News and Yahoo Questions指定`PATTERN_IDS 0 1 2 3 4`，为 Yelp Reviews and MNLI指定`PATTERN_IDS 0 1 2 3`。
 - `$DATA_DIR`  是包含训练和测试文件的目录（检查tasks.py以查看如何为每个任务命名和格式化这些文件）。
 - `$MODEL_TYPE`  是所使用模型的名称，例如`albert`，`bert`或`roberta`。
 - `$MODEL_NAME` 是预训练模型的名称（例如， `roberta-large` or `albert-xxlarge-v2`）或预训练模型的路径。
 - `$TASK_NAME`   是要进行训练和评估的任务的名称。
 - `$OUTPUT_DIR`  是保存经过训练的模型和评估结果的目录的名称。
 
您还可以为对应于各个PVP的PET模型集合（前缀`--pet_`）和最终序列分类模型（前缀`--sc_`）指定各种训练参数。
例如，用于我们的SuperGLUE评估的默认参数为：
 
 	--pet_per_gpu_eval_batch_size 8 \
	--pet_per_gpu_train_batch_size 2 \
	--pet_gradient_accumulation_steps 8 \
	--pet_max_steps 250 \
	--pet_max_seq_length 256 \
    --pet_repetitions 3 \
	--sc_per_gpu_train_batch_size 2 \
	--sc_per_gpu_unlabeled_batch_size 2 \
	--sc_gradient_accumulation_steps 8 \
	--sc_max_steps 5000 \
	--sc_max_seq_length 256 \
    --sc_repetitions 1
    
对于每个pattern `$P`和repetition `$I`，运行上面的命令将创建一个目录`$OUTPUT_DIR/p$P-i$I`，其中包含以下文件：
  - `pytorch_model.bin`: 经过微调的模型，可能还会包含一些特定于模型的文件(e.g, `spiece.model`, `special_tokens_map.json`)
  - `wrapper_config.json`: 正在使用的模型的配置
  - `train_config.json`: 用于训练的配置
  - `eval_config.json`: 用于评估的配置
  - `logits.txt`: 模型对无标签数据的预测
  - `eval_logits.txt`: 模型对评估数据的预测
  - `results.json`: ：一个包含结果的json文件，例如模型的最终精度
  - `predictions.jsonl`: SuperGlue格式的评估集的预测文件
  
每次epetition `$I` 的最终（蒸馏）模型可以在`$OUTPUT_DIR/final/p0-i$I`中找到，该模型包含与上述相同的文件。

🚨 如果您的GPU在训练期间内存不足，则可以尝试同时减 `pet_per_gpu_train_batch_size` and the `sc_per_gpu_unlabeled_batch_size`
 , 同时增加 `pet_gradient_accumulation_steps` and `sc_gradient_accumulation_steps`.


### iPET Training and Evaluation

要为其中一个任务训练和评估iPET模型，只需运行与上述相同的命令，
然后将`--method pet` 替换为`--method ipet`即可。您可以修改各种其他的iPET参数。它们都以`--ipet_`为前缀。

对于generation `$G`，pattern `$P` and iteration `$I`，
这将创建目录`$OUTPUT_DIR/g$G/p$P-i$I` ，其结构与常规PET相同。最终（提取的）模型可以再次在`$OUTPUT_DIR/final/p0-i$I`.中找到。

🚨如果将iPET与zero个训练示例一起使用，则需要指定在第一generation中应为每个标签选择多少个示例，
并且需要将减少策略更改为：`--ipet_n_most_likely 100 --reduction mean`.

### Supervised Training and Evaluation

要以监督的方式训练和评估常规序列分类器，只需运行与上述相同的命令，
但是将`--method pet` 替换为`--method sequence_classifier`即可。您可以修改序列分类器的各种其他参数。它们都以`--sc_`为前缀。

### Unsupervised Evaluation

要使用默认的PET pattern和verbalizers 评估经过预训练的语言模型，但无需进行微调，
请删除参数`--do_train`并添加`--no_distillation`，以便不执行最终的蒸馏。

### Priming

如果要使用priming，请删除参数 `--do_train` 并添加参数 `--priming --no_distillation`，以便所有训练样本均用于priming，并且不执行最终蒸馏。

🚨请记住，您可能需要将最大序列长度增加到更大的值，例如`--pet_max_seq_length 5000`。
这仅适用于支持此类长序列的语言模型，例如XLNet。为了使用XLNet，您可以指定`--model_type xlnet --model_name_or_path xlnet-large-cased --wrapper_type plm`。

## 💻 API Usage

除了使用命令行界面之外，您还可以直接使用PET API，其中大多数在`pet.modeling`中定义。
通过包含`import pet`，您可以访问诸如`train_pet`, `train_ipet` and `train_classifier`.之类的方法。查看他们的文档以获取更多信息。

## 🐶 Train your own PET

要将PET用于自定义任务，您需要定义两件事：

- a **DataProcessor**, 负责加载训练和测试数. See `examples/custom_task_processor.py` for an example.
- a **PVP**, 负责将pattern应用到输入并将标签映射到自然语言verbalizations. See `examples/custom_task_pvp.py` for an example.

在实现了DataProcessor和PVP之后，您可以如上所述使用命令行来训练PET模型[described above](#pet-training-and-evaluation)）。
在下面，您可以找到有关如何定义PVP的两个组件*verbalizers* and *patterns*的其他信息。

### Verbalizers

Verbalizers用于将任务标签映射到自然语言的单词。
例如，在二进制情感分类任务中，您可以将正标签（+1）映射到单词`good`，将负标签（-1）映射到`bad`。
Verbalizers是通过PVP的`verbalize()`方法实现的。定义verbalizer的最简单方法是使用字典：
```python
VERBALIZER = {"+1": ["good"], "-1": ["bad"]}
    
def verbalize(self, label) -> List[str]:
    return self.VERBALIZER[label]       
```
重要的是，在PET的当前版本中，默认情况下，verbalizers仅限于基础LM单词表中的 **single tokens** （要使用多个token，[see below](#pet-with-multiple-masks))。
给定语言模型的tokenizer，您可以通过验证`len(tokenizer.tokenize(word)) == 1`.来轻松检查单词是否与单个标签相对应。

您还可以为单个标签定义多个verbalizations。
例如，如果您不确定哪个词最能代表二进制情感分类任务中的标签，则可以按以下方式定义verbalizer：

```python
VERBALIZER = {"+1": ["great", "good", "wonderful", "perfect"], "-1": ["bad", "terrible", "horrible"]}
```

### Patterns

pattern用于使语言模型理解给定的任务。
它们必须只包含一个`<MASK>` token，该token将使用verbalizer填充。
对于基于评论摘要（`<A>`）和正文（`<B>`）的二进制情感分类，
合适的pattern可以是`<A>. <B>`。总的来说，它是<MASK>。
pattern是通过PVP的`get_parts()`方法实现的，该方法返回一对文本序列（每个序列由字符串列表表示）：

```python
def get_parts(self, example: InputExample):
    return [example.text_a, '.', example.text_b, '.'], ['Overall, it was ', self.mask]
```
如果您不想使用一对序列，则只需将第二个序列留空：

```python
def get_parts(self, example: InputExample):
    return [example.text_a, '.', example.text_b, '. Overall, it was ', self.mask], []
```

如果要定义几种pattern，只需使用PVP的pattern_id属性：            

```python
def get_parts(self, example: InputExample):
    if self.pattern_id == 1:
        return [example.text_a, '.', example.text_b, '.'], ['Overall, it was ', self.mask]
    elif self.pattern_id == 2:
        return ['It was just ', self.mask, '!', example.text_a, '.', example.text_b, '.'], []
```

使用命令行训练模型时，请指定要使用的所有pattern（例如，`--pattern_ids 1 2`).

重要的是，如果序列长于基础LM的指定最大序列长度，
则PET必须知道输入的哪些部分可以缩短而哪些部分不能缩短（例如，mask token必须始终存在）。
因此，PVP提供了`shortenable()` 方法来指示可以缩短一段文本：

```python
def get_parts(self, example: InputExample):
    text_a = self.shortenable(example.text_a)
    text_b = self.shortenable(example.text_b)
    return [text_a, '.', text_b, '. Overall, it was ', self.mask], []
```

### PET with Multiple Masks

默认情况下，PET和iPET的当前实现仅支持一组固定的标签，该标签在与单个token相对应的所有样本和verbalizers之间共享。
如果要使用对应于多个token的verbalizers, 如此处所述, http://arxiv.org/abs/2009.07118，
则需要定义一个自定义`TaskHelper`并将其添加到`TASK_HELPERS`字典中在`pet / tasks.py`中。
首先，您可以在`pet/task_helpers.py`中检出`CopaTaskHelper`, `WscTaskHelper` and `RecordTaskHelper` 类。
在下一版的PET中，默认情况下将启用带有多个masks的verbalizers。

## 📕 Citation

If you make use of the code in this repository, please cite the following papers:

    @article{schick2020exploiting,
      title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2001.07676},
      url={http://arxiv.org/abs/2001.07676},
      year={2020}
    }

    @article{schick2020small,
      title={It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2009.07118},
      url={http://arxiv.org/abs/2009.07118},
      year={2020}
    }
