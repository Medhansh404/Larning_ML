# input: 
1). What I plan to do is give numerical values (for now user gets to choose a number n)
that we pick the number of datapoints from a normal distribution.
2). For now only one type of optimiser and layer can be implemented(that'll be given to choose).
3). Stuff like number of epochs, learning rates, other constants, after what amt of iterations we wish to see
results

# API working:
1). it will give error, predicted value
2). suggestions for model like if image is uploaded use relu activation or similar type 
3). graphical interpretation of w and b in each model
4). processing of images and text 

## The basis is to appreciate the statistics and mathematics going in the core of shiny models by taking a closer 
## look at problems of different fields. 


# Thinking about the design approach: 2 options like pytorch, another is to make a simple approach
1). Pytorch: 
        pros: well, it is apparently the right way easily maintainable and gives structure, object-oriented
        cons: should know a lot before running a simple thing, complicated code, end up thinking about design and code nothing
2). Simple:
        pros: simple and does the job on simple stuff
        cons: not scalable and difficult to add stuff and make too many changes.