Recurrence Theory
=================

The key idea is that we assume that we are trying to model a sequence of some type.  When we say sequence, this could be a time ordered series of numbers or other symbols (like English text).  An important aspect is that the ordering (time or position in a sentence) has some meaning and is important to the overall model.  For example "the cat is over" evokes a meaning, whereas "eht tac si revo" (reversing each word in place) has no obvious meaning in English, even though the symbols are exactly the same in both sentences.

Recurrent Neural Networks attempt to model the sequential aspects of sentences like the first one above in order to, for example, predict which letter or word typically follows a given sequence.  Perhaps in the case above, the full sentence might be "the cat is over there near the fireplace" - if we were designing a completion algorithm for a search engine, it would be useful if the system knew that the completion (a common one) is "there near the fireplace."
