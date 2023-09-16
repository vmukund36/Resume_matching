# Resume Matching with Textract and DistilBERT
This project demonstrates how to build a resume matching system to compare job descriptions with candidate resumes. We utilize Textract for extracting text from PDF resumes and the DistilBERT model to calculate similarity scores. The goal is to identify the most relevant resumes for a given job description. <br /> 
For this task, I used the first 15 job descriptions from the HuggingFace dataset. (Link to dataset: https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/viewer/default/train?row=0)
## Model Description
DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model. More precisely, it was pretrained with three objectives:<br />

1) Distillation loss: the model was trained to return the same probabilities as the BERT base model.
2) Masked language modeling (MLM): this is part of the original training loss of the BERT base model. When taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.
3) Cosine embedding loss: the model was also trained to generate hidden states as close as possible as the BERT base model.
This way, the model learns the same inner representation of the English language than its teacher model, while being faster for inference or downstream tasks.

## Textract 
Textract is considered a useful tool for text extraction, particularly from PDFs and other document formats, due to several advantages:

1. Support for Multiple Formats: Textract supports a wide range of document formats, including PDF, DOC, DOCX, XLSX, PPTX, and more. This versatility makes it a valuable choice for extracting text from various types of documents.

2. Simplicity: Textract provides a simple and straightforward interface for extracting text. You can typically extract text from a document with just a few lines of code, making it accessible for both beginners and experienced developers.

3. Accuracy: Textract is designed to extract text accurately, preserving the formatting and structure of the original document. It can handle complex documents with tables, images, and multiple fonts.

4. Platform Independence: Textract is available for multiple programming languages, including Python, Node.js, and Java, making it versatile and suitable for a wide range of development environments.

5. Open Source: Textract is open source and freely available, which means you can use it without incurring additional costs. This open-source nature also encourages community contributions and improvements.

6. Customization: Textract allows you to customize the extraction process to some extent. You can specify options for handling document-specific features or configuring the extraction behavior.

7. Cross-Platform Compatibility: Textract can be used on various operating systems, including Windows, macOS, and Linux, ensuring compatibility with different development environments.

8. Integration with Other Libraries: Textract can be easily integrated with other libraries and tools commonly used in data processing and analysis pipelines, such as natural language processing (NLP) libraries or machine learning frameworks.

9. Scalability: Textract can be applied to process a large number of documents efficiently, making it suitable for tasks involving large document collections or document management systems.

10. Community and Support: Due to its popularity and open-source nature, Textract benefits from an active community of users and contributors. You can find documentation, tutorials, and community support to assist with your projects.
    
## Prerequisites
Before you get started, ensure you have the following dependencies installed: <br />

* Python (3.9)
* PyTorch (for the DistilBERT model)
* Transformers library (for DistilBERT)
* Textract (for PDF text extraction)
* scikit-learn (for cosine similarity)
  ```python
  pip install torch transformers textract numpy scikit-learn
  ```

<br />
Alternatively, these can be installed using `Requirements.txt`
<br />

## USAGE
1. Clone the Repository
```python
 git clone https://github.com/vmukund36/Resume_matching.git
```
2. Dataset Preperation
   <br />
   <br />
   Download the dataset using the link: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset. Change the dataset/folder path in the code as per your local path. 
   <br />
   <br />
3. Running the script 
```python
 python3 main.py
```
<br />
This script will process the job descriptions, extract text from resumes, calculate similarity scores, and generate a list of top matching resumes for each job description.
<br />
<br />
4. Review Results
<br />
The results will be displayed in the terminal, showing the top 5 matching resumes for each specified job description along with similarity scores. In the python notebook `Capital_placement_assignment.ipynb`, I have chosen 'Accountant' for testing the CV matching and it produced good results and displayed top 5 relevant resumés.
