from prompt_function import PromptFunction

# define the function
find_country = PromptFunction("Which country is this city located in? Output ONLY the Country name", 
                          max_tokens=5, model=0)

# call the function, after loading, each call should be fairly fast
print(f'Vancouver is located in {find_country("Vancouver")}')
print(f'Toronto is located in {find_country("Toronto")}')
print(f'San Francisco is located in {find_country("San Francisco")}')
print(f'Hong Kong is located in {find_country("Hong Kong")}')
