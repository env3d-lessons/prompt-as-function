from prompt_function import PromptFunction

# define the function
find_country = PromptFunction("Which country is this city located?  Output ONLY the Country name ", 
                          max_tokens=5, model=0)

# call the function, after loading, each call should be fairly fast
print(find_country("Vancouver"))
print(find_country("Toronto"))
print(find_country("San Francisco"))
print(find_country("Hong Kong"))
