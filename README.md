# ocr-from-historic-documents
Optical character recognition (OCR) for historic printed documents using convolutional neural networks

Identify (classify) characters from bitmap images extracted from historical texts

Optical Character Recognition (OCR) is an important machine learning application which converts an image of a printed document into text. This is a useful tool for digitising vast collections of books such as in Iibraries, meaning that text can be used efficiently for efficient searching and other applications. While OCR for modern documents is largely a solved problem, it remains very challenging case for historical documents. Early printing presses required manual selection and placement of each of the characters by the printer, and features wandering baselines (horizontal lines on which the letters "sit"), ink splodges, use of odd fonts and calligraphic capitals, and the use of characters that are no longer in use in modern texts. These issues mean that off the shelf application of modern OCR technologies produce mostly gibberish, and consequently is of little practical use.

In this project you will develop character classifiers for several historical documents, each produced shortly after the advent of the printing press. Note that these documents are in different languages, use different fonts, and have other idiosyncrasies specific to their author and printer. Your job is to identify for a given bitmap image of a character the identity of that character.
