<html>
<head>
<title>VisualCortexBENN.py</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #7a7e85;}
.s1 { color: #bcbec4;}
.s2 { color: #cf8e6d;}
.s3 { color: #bcbec4;}
.s4 { color: #2aacb8;}
.s5 { color: #6aab73;}
</style>
</head>
<body bgcolor="#1e1f22">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
VisualCortexBENN.py</font>
</center></td></tr></table>
<pre><span class="s0">#   Title:  VisualCortexBENN</span>
<span class="s0">#   Desc:   A program that allows for better image processing</span>
<span class="s0">#           utilizing a more incentive reward system.</span>
<span class="s0">#   Author: Angela Trainor</span>
<span class="s0">#   Date:   08/22/2025</span>

<span class="s2">import </span><span class="s1">torch</span>

<span class="s2">import </span><span class="s1">torch</span><span class="s3">.</span><span class="s1">nn </span><span class="s2">as </span><span class="s1">nn</span>

<span class="s2">import </span><span class="s1">torch</span><span class="s3">.</span><span class="s1">nn</span><span class="s3">.</span><span class="s1">functional </span><span class="s2">as </span><span class="s1">F</span>



<span class="s2">class </span><span class="s1">VisualCortexBENN</span><span class="s3">(</span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Module</span><span class="s3">):</span>

    <span class="s2">def </span><span class="s1">__init__</span><span class="s3">(</span><span class="s1">self</span><span class="s3">):</span>

        <span class="s1">super</span><span class="s3">().</span><span class="s1">__init__</span><span class="s3">()</span>



        <span class="s0"># 🎨 Low-Level Feature Extraction (V1–V3 simulation)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">conv1 </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Conv2d</span><span class="s3">(</span><span class="s4">3</span><span class="s3">, </span><span class="s4">32</span><span class="s3">, </span><span class="s1">kernel_size</span><span class="s3">=</span><span class="s4">3</span><span class="s3">, </span><span class="s1">padding</span><span class="s3">=</span><span class="s4">1</span><span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">conv2 </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Conv2d</span><span class="s3">(</span><span class="s4">32</span><span class="s3">, </span><span class="s4">64</span><span class="s3">, </span><span class="s1">kernel_size</span><span class="s3">=</span><span class="s4">3</span><span class="s3">, </span><span class="s1">padding</span><span class="s3">=</span><span class="s4">1</span><span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">conv3 </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Conv2d</span><span class="s3">(</span><span class="s4">64</span><span class="s3">, </span><span class="s4">128</span><span class="s3">, </span><span class="s1">kernel_size</span><span class="s3">=</span><span class="s4">3</span><span class="s3">, </span><span class="s1">padding</span><span class="s3">=</span><span class="s4">1</span><span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">pool </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">AdaptiveAvgPool2d</span><span class="s3">((</span><span class="s4">8</span><span class="s3">, </span><span class="s4">8</span><span class="s3">))  </span><span class="s0"># Preserve spatial layout</span>



        <span class="s0"># 💓 Color &amp; Texture Embedding</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">color_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">256</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">(),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">256</span><span class="s3">, </span><span class="s4">64</span><span class="s3">)</span>

        <span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">texture_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">256</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">(),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">256</span><span class="s3">, </span><span class="s4">64</span><span class="s3">)</span>

        <span class="s3">)</span>



        <span class="s0"># 🌀 Shape Emotion Mapping</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">symmetry_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">64</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">(),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">64</span><span class="s3">, </span><span class="s4">8</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Sigmoid</span><span class="s3">()</span>

        <span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">curvature_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">64</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">(),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">64</span><span class="s3">, </span><span class="s4">8</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Sigmoid</span><span class="s3">()</span>

        <span class="s3">)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">complexity_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">64</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">(),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">64</span><span class="s3">, </span><span class="s4">8</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Sigmoid</span><span class="s3">()</span>

        <span class="s3">)</span>



        <span class="s0"># 📍 Centroid Estimation (for spatial matching)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">centroid_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">32</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Tanh</span><span class="s3">()  </span><span class="s0"># Normalized spatial coordinates</span>

        <span class="s3">)</span>



        <span class="s0"># 🧬 Entity Classifier (optional expansion)</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">entity_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">128</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">ReLU</span><span class="s3">()</span>

        <span class="s3">)</span>



        <span class="s0"># 🧠 Attention Gating</span>

        <span class="s1">self</span><span class="s3">.</span><span class="s1">attention_fc </span><span class="s3">= </span><span class="s1">nn</span><span class="s3">.</span><span class="s1">Sequential</span><span class="s3">(</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Linear</span><span class="s3">(</span><span class="s4">128 </span><span class="s3">* </span><span class="s4">8 </span><span class="s3">* </span><span class="s4">8</span><span class="s3">, </span><span class="s4">1</span><span class="s3">),</span>

            <span class="s1">nn</span><span class="s3">.</span><span class="s1">Sigmoid</span><span class="s3">()</span>

        <span class="s3">)</span>



    <span class="s2">def </span><span class="s1">forward</span><span class="s3">(</span><span class="s1">self</span><span class="s3">, </span><span class="s1">image</span><span class="s3">):</span>

        <span class="s0"># 🎨 Feature Extraction</span>

        <span class="s1">x </span><span class="s3">= </span><span class="s1">F</span><span class="s3">.</span><span class="s1">relu</span><span class="s3">(</span><span class="s1">self</span><span class="s3">.</span><span class="s1">conv1</span><span class="s3">(</span><span class="s1">image</span><span class="s3">))</span>

        <span class="s1">x </span><span class="s3">= </span><span class="s1">F</span><span class="s3">.</span><span class="s1">relu</span><span class="s3">(</span><span class="s1">self</span><span class="s3">.</span><span class="s1">conv2</span><span class="s3">(</span><span class="s1">x</span><span class="s3">))</span>

        <span class="s1">x </span><span class="s3">= </span><span class="s1">F</span><span class="s3">.</span><span class="s1">relu</span><span class="s3">(</span><span class="s1">self</span><span class="s3">.</span><span class="s1">conv3</span><span class="s3">(</span><span class="s1">x</span><span class="s3">))</span>

        <span class="s1">pooled </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">pool</span><span class="s3">(</span><span class="s1">x</span><span class="s3">)  </span><span class="s0"># (B, 128, 8, 8)</span>

        <span class="s1">flat </span><span class="s3">= </span><span class="s1">pooled</span><span class="s3">.</span><span class="s1">view</span><span class="s3">(</span><span class="s1">pooled</span><span class="s3">.</span><span class="s1">size</span><span class="s3">(</span><span class="s4">0</span><span class="s3">), -</span><span class="s4">1</span><span class="s3">)  </span><span class="s0"># (B, 8192)</span>



        <span class="s0"># 💓 Emotional Embeddings</span>

        <span class="s1">color_embed </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">color_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)</span>

        <span class="s1">texture_embed </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">texture_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)</span>



        <span class="s1">symmetry </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">symmetry_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)</span>

        <span class="s1">curvature </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">curvature_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)</span>

        <span class="s1">complexity </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">complexity_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)</span>

        <span class="s1">shape_emotion </span><span class="s3">= </span><span class="s1">symmetry </span><span class="s3">+ </span><span class="s1">curvature </span><span class="s3">+ </span><span class="s1">complexity  </span><span class="s0"># (B, 8)</span>



        <span class="s0"># 📍 Centroid for spatial matching</span>

        <span class="s1">centroid </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">centroid_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)  </span><span class="s0"># (B, 32)</span>



        <span class="s0"># 🧬 Entity Embedding</span>

        <span class="s1">entity_embed </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">entity_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)  </span><span class="s0"># (B, 128)</span>



        <span class="s0"># 🧠 Attention Score</span>

        <span class="s1">attention </span><span class="s3">= </span><span class="s1">self</span><span class="s3">.</span><span class="s1">attention_fc</span><span class="s3">(</span><span class="s1">flat</span><span class="s3">)  </span><span class="s0"># (B, 1)</span>



        <span class="s2">return </span><span class="s3">{</span>

            <span class="s5">&quot;color&quot;</span><span class="s3">: </span><span class="s1">color_embed</span><span class="s3">,</span>

            <span class="s5">&quot;texture&quot;</span><span class="s3">: </span><span class="s1">texture_embed</span><span class="s3">,</span>

            <span class="s5">&quot;shape_emotion&quot;</span><span class="s3">: </span><span class="s1">shape_emotion</span><span class="s3">,</span>

            <span class="s5">&quot;centroid&quot;</span><span class="s3">: </span><span class="s1">centroid</span><span class="s3">,</span>

            <span class="s5">&quot;entities&quot;</span><span class="s3">: </span><span class="s1">entity_embed</span><span class="s3">,</span>

            <span class="s5">&quot;attention&quot;</span><span class="s3">: </span><span class="s1">attention</span>

        <span class="s3">}</span>


</pre>
</body>
</html>