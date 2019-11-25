"""
Custom Module with common tools for use in Notebooks 

Dan Boschen 11/28/2018
"""
import itertools as it

# demonstrating importing a module global variable
x = 5

def disp(my_list, ncol = 4, width = 20):
    """
    Display list in column format, each successive item in list
    will be displayed in a separate column (increments by col
    first and then row)
    
    Dan Boschen 11/25/2019
    
    Parameters:
    -----------
    
    my_list:  List
    
    ncol: integer, optional
        number of columns, default = 4
        
    width: width, optional
        column spacing, default = 20
        
    Returns:
    --------
    None
    
    """
    def _abrev(name, width):
        if len(str(name)) > width-4:
            return str(name)[:width-4] + "..."
        else:
            return str(name)
        
    # ensure string and shorten all items to column width
    my_new_list = [_abrev(i, width) for i in my_list ]


    #create a format string for ncol columns spaced by width
    # result for width = 20 and ncol = 4 is  
    # "{:<20}{:<20}{:<20}{:<20}"
    template = "".join(("{:<"+str(width)+"}") * ncol) 
    
    #print using template
    for columns in  it.zip_longest(*[iter(my_new_list)] * ncol, fillvalue = ""):
        print(template.format(*columns))
        
        
def printmd(stringWithMarkdown):
    """
    Prints string with Markdown as an output of a code block
    """
    from IPython.display import Markdown, display
    display(Markdown(stringWithMarkdown))
