# 记录paper撰写的流程

使用工具：
- latex: 使用平台[overleaf](https://cn.overleaf.com/project)
- 图片转latex: https://snip.mathpix.com/747876457/notes/note-1/edit
- latex转图片: https://www.latexlive.com/

如何开始撰写论文？

- 参考: https://github.com/secdr/research-method, 阅读部分如下
  - [如何写好一篇论文](https://github.com/secdr/research-method/blob/master/how%20to%20write/%E5%A6%82%E4%BD%95%E5%86%99%E5%A5%BD%E4%B8%80%E7%AF%87%E8%AE%BA%E6%96%87.pdf) 
  - [latex撰写伪代码](https://github.com/secdr/research-method/blob/master/how%20to%20write/%E7%94%A8LaTeX%E4%BC%98%E9%9B%85%E5%9C%B0%E4%B9%A6%E5%86%99%E4%BC%AA%E4%BB%A3%E7%A0%81%E2%80%94%E2%80%94Algorithm2e%E7%AE%80%E6%98%8E%E6%8C%87%E5%8D%97%20-%20%E7%9F%A5%E4%B9%8E.pdf) 

如何使用latex？

- kdd latex模板: https://www.kdd.org/author-instructions，需要下载两个文件.tex
  & .cls文件
- 使用软件：[overleaf](https://cn.overleaf.com) , 注意在设置中将texlive版本改
  成2020，然后编译成pdf
- 参考：[简单粗暴latex](https://github.com/wklchris/Note-by-LaTeX)

**参考文献管理方法**:

1. 创建bib文件，使用Google scholar 查找文献，点击下面的引用，复制bibtex格式，
   粘贴到bib文件中，bib文件要和tex一样的目录
2. 在`\documentclass{main}` 下面添加引用相关的包`\usepackage{cite}` 
3. bib文件中，有每个参考文献的唯一标识符，在tex文件中使用`\cite{ref1}` 引用这
   个论文

**公式** :

使用下面方式，会自带序号, 使用label为了方便引用：

```
\begin{equation}\label{1}

\end{equation}
```

引用的时候：

```
\eqref{1}
```

内容：

- paper.md：论文的中文内容，顺序为先写中文，然后再写英文
