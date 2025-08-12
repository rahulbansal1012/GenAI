from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"
)

llm2  =  HuggingFaceEndpoint(
     repo_id= "Qwen/Qwen3-Coder-480B-A35B-Instruct",task= "text-generation"
    
)

model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)

template1 =  PromptTemplate(
    template= "You task is to Generate the notes on the following content \n {text}",
    input_variables=['text']
)

tempalte2 =  PromptTemplate(
    template= "Your task is to generate 5 Quiz on the Following text \n {text}",
    input_variables=['text']
)

tempalte3 = PromptTemplate(
    template= "your task is to combine both the notes and quiz question in a single document \n notes: {notes} \n quiz : {quiz}",
    input_variables= ['notes' , 'quiz']
)
parser =  StrOutputParser()
# generating notes there

parallel_chain = RunnableParallel(
    {'notes' : template1 | model1 | parser ,
     'quiz' : tempalte2 | model2 |parser
     }
)

merge_chain = tempalte3 | model1 | parser

resulting_chain  = parallel_chain | merge_chain

text = '''n machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised max-margin models with associated learning algorithms that analyze data for classification and regression analysis. Developed at AT&T Bell Laboratories,[1][2] SVMs are one of the most studied models, being based on statistical learning frameworks of VC theory proposed by Vapnik (1982, 1995) and Chervonenkis (1974).

In addition to performing linear classification, SVMs can efficiently perform non-linear classification using the kernel trick, representing the data only through a set of pairwise similarity comparisons between the original data points using a kernel function, which transforms them into coordinates in a higher-dimensional feature space. Thus, SVMs use the kernel trick to implicitly map their inputs into high-dimensional feature spaces, where linear classification can be performed.[3] Being max-margin models, SVMs are resilient to noisy data (e.g., misclassified examples). SVMs can also be used for regression tasks, where the objective becomes 

{\displaystyle \epsilon }-sensitive.

The support vector clustering[4] algorithm, created by Hava Siegelmann and Vladimir Vapnik, applies the statistics of support vectors, developed in the support vector machines algorithm, to categorize unlabeled data.[citation needed] These data sets require unsupervised learning approaches, which attempt to find natural clustering of the data into groups, and then to map new data according to these clusters.

The popularity of SVMs is likely due to their amenability to theoretical analysis, and their flexibility in being applied to a wide variety of tasks, including structured prediction problems. It is not clear that SVMs have better predictive performance than other linear models, such as logistic regression and linear regression.[5]

Motivation

H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximal margin.
Classifying data is a common task in machine learning. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a 
p
{\displaystyle p}-dimensional vector (a list of 
p
{\displaystyle p} numbers), and we want to know whether we can separate such points with a 
(
p
−
1
)
{\displaystyle (p-1)}-dimensional hyperplane. This is called a linear classifier. There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two classes. So we choose the hyperplane so that the distance from it to the nearest data point on each side is maximized. If such a hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier it defines is known as a maximum-margin classifier; or equivalently, the perceptron of optimal stability.[6]

More formally, a support vector machine constructs a hyperplane or set of hyperplanes in a high or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.[7] Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier.[8] A lower generalization error means that the implementer is less likely to experience overfitting.


Kernel machine
Whereas the original problem may be stated in a finite-dimensional space, it often happens that the sets to discriminate are not linearly separable in that space. For this reason, it was proposed[9] that the original finite-dimensional space be mapped into a much higher-dimensional space, presumably making the separation easier in that space. To keep the computational load reasonable, the mappings used by SVM schemes are designed to ensure that dot products of pairs of input data vectors may be computed easily in terms of the variables in the original space, by defining them in terms of a kernel function 
k
(
x
,
y
)
{\displaystyle k(x,y)} selected to suit the problem.[10] The hyperplanes in the higher-dimensional space are defined as the set of points whose dot product with a vector in that space is constant, where such a set of vectors is an orthogonal (and thus minimal) set of vectors that defines a hyperplane. The vectors defining the hyperplanes can be chosen to be linear combinations with parameters 
α
i
{\displaystyle \alpha _{i}} of images of feature vectors 
x
i
{\displaystyle x_{i}} that occur in the data base. With this choice of a hyperplane, the points 
x
{\displaystyle x} in the feature space that are mapped into the hyperplane are defined by the relation 
∑
i
α
i
k
(
x
i
,
x
)
=
constant
.
{\displaystyle \textstyle \sum _{i}\alpha _{i}k(x_{i},x)={\text{constant}}.} Note that if 
k
(
x
,
y
)
{\displaystyle k(x,y)} becomes small as 
y
{\displaystyle y} grows further away from 
x
{\displaystyle x}, each term in the sum measures the degree of closeness of the test point 
x
{\displaystyle x} to the corresponding data base point 
x
i
{\displaystyle x_{i}}. In this way, the sum of kernels above can be used to measure the relative nearness of each test point to the data points originating in one or the other of the sets to be discriminated. Note the fact that the set of points 
x
{\displaystyle x} mapped into any hyperplane can be quite convoluted as a result, allowing much more complex discrimination between sets that are not convex at all in the original space.

Applications
SVMs can be used to solve various real-world problems:

SVMs are helpful in text and hypertext categorization, as their application can significantly reduce the need for labeled training instances in both the standard inductive and transductive settings.[11] Some methods for shallow semantic parsing are based on support vector machines.[12]
Classification of images can also be performed using SVMs. Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback. This is also true for image segmentation systems, including those using a modified version SVM that uses the privileged approach as suggested by Vapnik.[13][14]
Classification of satellite data like SAR data using supervised SVM.[15]
Hand-written characters can be recognized using SVM.[16][17]
The SVM algorithm has been widely applied in the biological and other sciences. They have been used to classify proteins with up to 90% of the compounds classified correctly. Permutation tests based on SVM weights have been suggested as a mechanism for interpretation of SVM models.[18][19] Support vector machine weights have also been used to interpret SVM models in the past.[20] Posthoc interpretation of support vector machine models in order to identify features used by the model to make predictions is a relatively new area of research with special significance in the biological sciences.
History
The original SVM algorithm was invented by Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1964.[citation needed] In 1992, Bernhard Boser, Isabelle Guyon and Vladimir Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes.[9] The "soft margin" incarnation, as is commonly used in software packages, was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.[1]'''

output  = resulting_chain.invoke({'text' : text})
print(output)
print(resulting_chain.get_graph().draw_ascii()    )
