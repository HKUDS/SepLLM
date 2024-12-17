# PG-19 Language Modelling Benchmark
This repository contains the PG-19 language modeling benchmark. It includes a
set of books extracted rom the Project Gutenberg books library [1], that were
published before 1919. It also contains metadata of book titles and publication
dates.

<b><a href="https://console.cloud.google.com/storage/browser/deepmind-gutenberg">Full dataset download link</a></b>

PG-19 is over double the size of the Billion Word benchmark [2] and contains
documents that are 20X longer, on average, than the WikiText long-range language
modelling benchmark [3].

Books are partitioned into a `train`, `validation`, and `test` set. Book
metadata is stored in `metadata.csv` which contains
`(book_id, short_book_title, publication_date)`.

Unlike prior benchmarks, we do not constrain the vocabulary size ---
i.e. mapping rare words to an UNK token --- but instead release the data as an
open-vocabulary benchmark. The only processing of the text that has been applied
is the removal of boilerplate license text, and the mapping of offensive
discriminatory words as specified by Ofcom [4] to placeholder <DW> tokens. Users
are free to model the data at the character-level, subword-level, or via any
mechanism that can model an arbitrary string of text.

To compare models we propose to continue measuring the word-level perplexity,
by calculating the total likelihood of the dataset (via any chosen subword
vocabulary or character-based scheme) divided by the number of tokens ---
specified below in the dataset statistics table.

One could use this dataset for benchmarking long-range language models, or
use it to pre-train for other natural language processing tasks which require
long-range reasoning, such as LAMBADA [5] or NarrativeQA [6]. We would not
recommend using this dataset to train a general-purpose language model, e.g.
for applications to a production-system dialogue agent, due to the dated
linguistic style of old texts and the inherent biases present in historical
writing.

### Dataset Statistics

<table >
    <tbody>
        <tr>
            <td> </td>
            <td> Train </td>
            <td> Validation </td>
            <td> Test </td>
        </tr>
        <tr>
            <td> Books </td>
            <td> 28,602 </td>
            <td> 50 </td>
            <td> 100 </td>
        </tr>
        <tr>
            <td>Num. Tokens </td>
            <td> 1,973,136,207 </td>
            <td> 3,007,061 </td>
            <td> 6,966,499 </td>
        </tr>
    </tbody>
</table>

### Bibtex

```
@article{raecompressive2019,
author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and
          Hillier, Chloe and Lillicrap, Timothy P},
title = {Compressive Transformers for Long-Range Sequence Modelling},
journal = {arXiv preprint},
url = {https://arxiv.org/abs/1911.05507},
year = {2019},
}
```

### Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">The PG-19 Language Modeling Benchmark</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">PG-19</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/pg19</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/pg19</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">This repository contains the PG-19 dataset.
    It includes a set of books extracted from the Project Gutenberg
    books project (https://www.gutenberg.org), that were published before
    1919. It also contains metadata of book titles and publication dates.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Apache License, Version 2.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://www.apache.org/licenses/LICENSE-2.0.html</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://identifiers.org/arxiv:1911.05507</code></td>
  </tr>
</table>
</div>

### Contact

If you have any questions, please contact <a href="mailto:jwrae@google.com">Jack Rae</a>.

### References

<ul style="list-style: none;">
<li>[1] <a href="https://www.gutenberg.org/">https://www.gutenberg.org</a></li>
<li>[2] Chelba et al. "One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling" (2013)</li>
<li>[3] Merity et al. "Pointer Sentinel Mixture Models" (2016)</li>
<li>[4] <a href="https://www.ofcom.org.uk/__data/assets/pdf_file/0023/91625/OfcomQRG-AOC.pdf">Ofcom offensive language guide</a></li>
<li>[5] Paperno et al. "The LAMBADA dataset: Word prediction requiring a broad discourse context" (2016)</li>
<li>[6] Kočiský et al. "The narrativeqa reading comprehension challenge" (2018)</li>
</ul>


