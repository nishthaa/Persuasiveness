;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;
;; CORPS - Release 2   ;;
;;   January 2011      ;;
;; FBK-Irst and CELCT  ;; 
;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;


CORPS is a corpus of political speeches tagged with specific audience reactions, such as APPLAUSE or LAUGHTER. 


------- CONTENT OF DIRECTORY CORPS_II_RELEASE ------

./data

This directory contains:

- the file CORPS_I.zip is the first release of CORPS "corpus_political_speeches-866" (this version is superseded by the new one). 
- the directory CORPS_II with the new release of CORPS. This includes also the first release, revised, plus the new speeches annotated by CELCT. 
  All the files are in UTF-8 encoding. Usually the file name is composed by speaker name plus the date of speech delivery. E.g. "gwbush21-11-03.txt".


./docs

This directory contains various pieces of information about CORPS II:

- The file statistics.txt, with main statistics about CORPS II.
- The file focus.xls, with the statistics about focus (Positive-Focus, Negative-Focus and Ironical-Focus tags). For further details see Guerini et al., (2008).
- The file audience.txt, with the complete dump of the tag {AUDIENCE}.
- The file comments.txt, with the complete dump of the tag {COMMENT}.
- The file mappingCompleteTab.xls, with the mapping of file names between CORPS I and CORPS II.



------- CORPS II FILE STRUCTURE ---------------------

The structure of the files, containing speech transcriptions, is as follow: 

{title} [mandatory - describing the speech] {/title}
{event} [not mandatory - derivable from the title most of the times] {/event} 
{speaker} [mandatory] {/speaker}
{date} [mandatory] {/date}
{source} [mandatory - internet address] {/source}

{description} [not mandatory - put if present in the source] {/description}

{speech} [speech transcription with the tags, see next section] {/speech}



------- TAGS DESCRIPTION ----------------------------

The speeches have been collected from internet, and a semi-automatic conversion of audience reactions tags has been performed to make them homogeneous. 

- Main tags in speech transcription are: 

{APPLAUSE}
{LAUGHTER}

- Other tags used: 

{SPONTANEOUS-DEMONSTRATION} 
tags replaced --> "reaction" "audience interruption"

{STANDING-OVATION}

{SUSTAINED APPLAUSE} 
tags replaced --> "extensive applause" "big applause" "loud applause" etc.

{CHEERS} = cries or shouts of approval from the audience as a whole (the audience roars) 
tags replaced --> "cries" "shouts" "whistles" etc.

{BOOING}
tags replaced --> "boo" "hissing" etc.

- Special tags:

{OTHER-SPEAK} [text] {/OTHER-SPEAK} 

This tag is used to signal speakers other than the subject (like journalists asking questions, chairmen, etc.). If this tag contains long texts and/or it is present many times in the speech transcription, we can infer that it is a semidialogical or dialogical situation.  


{AUDIENCE} [text] {/AUDIENCE} 

This tag is used to signal audience's intervention. It is not subsumed in the {OTHER-SPEAK} tag because it is typical for monological situations. Moreover in analysis it can be used in a way that is more similar to the use of "applause" and "cheers" tags. 


{AUDIENCE-MEMBER} [text] {/AUDIENCE-MEMBER}

This tag is used to signal a single audience member's intervention. It is not subsumed in the {OTHER-SPEAK} tag. 


- In the case of multiple tagging at a given point of the speech the used notation is: 

{[tag1] ; [tag2] ; ...}

Usually there are at most two tags. Obviously, in case of multiple tagging, tag1 must be different from tag2.

- The tag "comment" is used for special cases, e.g. :

{COMMENT = "A moment of silence was observed"} 
{COMMENT = "An audience member claps"} 
{COMMENT = "mentioning some names"} 
{COMMENT = "recording interrupted"}
{COMMENT = "inaudible"}



------- REFERENCES ----------------------------

Whenever making reference to this resource please cite one of the following papers.

For a general reference to the resource and qualitative analysis:
Guerini M., Strapparava C. & Stock O. "CORPS: A Corpus of Tagged Political Speeches for Persuasive Communication Processing". Journal of Information Technology & Politics, 5(1): 19-32, Routledge, 2008.

For specific automatic recognition of persuasion:
Strapparava C., Guerini M. & Stock O. "Predicting Persuasiveness in Political Discourses". In Proceedings of LREC2010, May 2010.
