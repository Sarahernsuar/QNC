# Sara Hernandez HW 1

#Part 1
print("If someone gets a positive test, is it statistically significant at the p=0.05 level? Why or why not")
#Set Variables
Sample = 1000
FalsePos = 0.05
TruePos = 1.00
ProbHIV = 1/Sample
ProbHealth = 1-ProbHIV

#Define probability using law of total probability
ProbPos = TruePos*ProbHIV + FalsePos*ProbHealth

pvalue = 0.05

#Answer
if ProbPos <= pvalue:
    print("It is significant")
    print(ProbPos, "is less or equal than", pvalue)
else:
    print("It is not significant")
    print("The probability of receiving a positive result is",ProbPos, ", which is greater than", pvalue )

## Part 2
print("What is the probability that if someone gets a positive test, that person is infected")

Health = 1.0
Sick = 0.0
pTruePos = 1.0
pFalsePos = 0.05
#Probability of a true Positive...
#p(HIV|pos)= p(pos|HIV)*p(HIV)/p(pos)
#where...
#Prob of positive -> p(pos)=p(pos|HIV)*p(HIV)+p(pos|healthy)*p(healthy)

while Health >= 0:
    print("Health: ",Health)
    pPositive = pTruePos*Sick+pFalsePos*Health
    #print(pPositive)
    pHIVgivenPositive = pTruePos*Sick/pPositive
    print("Probability of HIV given positive test: ", pHIVgivenPositive)
    Health = Health-0.1
    Sick = Sick+0.1