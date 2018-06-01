*****************************************************************************
* Annotated by: Qian Liu, Bing Liu, 2015.
* School of Computer Science and Engineering, Southeast University, China
* Department of Computer Science, University of Illinois at Chicago, USA
*
* Contact: Qian Liu, judy.liuqian@gmail.com or qianliu@seu.edu.cn
*          Bing Liu, liub@cs.uic.edu (http://www.cs.uic.edu/~liub)
*****************************************************************************

                            Readme file

This folder contains annotated customer reviews of 3 products:

1. Computer (531 sents): includes two files, i.e., Computer.xml and ComputerSents.txt
2. Wirless router (879 sents): incluses two files, i.e., Router.xml and RouterSents.txt
3. Speaker (689 sents): includes two files, i.e., Speaker.xml and SpeakerSents.txt

All the reviews were from amazon.com. They were used in the following paper:

Qian Liu, Zhiqiang Gao, Bing Liu, Yuanlin Zhang. Automated Rule Selection for Aspect Extraction. In Proceedings of IJCAI '15, 2015. 

Symbols used in the annotated reviews:

Exmaple 1: 

slower[-1][a], screen quality[+1], hard drive[+1] ## It is slightly slower than the dell , but it is hard to notice with the very nice screen quality and larger hard drive .

	1. xxxx[+|-1]: xxxx is a product feature/aspect. 
			 [+1]: Positive opinion
             [-1]: Negative opinion (comparative in this case)
	2. ##: start of each sentence. Each line is a sentence.
	3. [a]: an adjective which can imply a product feature/aspect, i.e., implicit feature/aspect (probably an opinion word) that is not appeared in the sentence. For example, expensive (implies price), beautiful (implies appearance).
	4. [v] : a verb feature which can imply a product feature/aspect, e.g., install (implies installation).
	
 
Example 2:
  <sentence id="26">
		<text>It is slightly slower than the dell , but it is hard to notice with the very nice screen quality and larger hard drive .</text>
		<aspectTerms>
			<aspectTerm from="15" to="20" polarity="negative" term="slower" pos="jj"/>
			<aspectTerm from="82" to="95" polarity="positive" term="screen quality" pos="nn"/>
			<aspectTerm from="108" to="117" polarity="positive" term="hard drive" pos="nn"/>
		</aspectTerms>
	</sentence>

	1. <sentence id="XX"></sentence>: the annotation for sentence XX.
	2. <text></text>: the text of sentence XX.
	3. <aspectTerms></aspectTerms>: the annotated features/aspects in sentence XX.
	4. <aspectTerm from="YY" to="ZZ" polarity="XXX" term="YYY" pos="ZZZ"/>: attributes of the feature/aspect YYY, YY (ZZ) is the start (end) index of term YYY in sentence XX, pos stands for the part of speech of YYY. nn is noun, jj is adjective, vb is verb.


