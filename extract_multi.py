import sys
from prompt_function import PromptFunction, timer

# Parse model index from command-line argument
model_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

extract_name = PromptFunction("Extract the person's full name from the following sentence, output only name:",
                              model = model_index, max_tokens=5)
extract_date = PromptFunction("Extract the date mentioned in the following sentence, output only date in the format YYYY-MM-DD:",
                              model = model_index, max_tokens=20)

test_inputs = [
    "Emily Zhang arrived in Montreal on March 15, 2021 for the conference.",
    "We had dinner with Carlos Rivera on August 3rd, 2023 in Vancouver.",
    "Sophie Dubois moved to Toronto back on July 1, 2020.",
    "I met Arjun Patel during the hackathon on November 12th, 2022.",
    "Li Wei submitted her thesis on May 9, 2019.",
    "Michael Thompson's flight was delayed on February 28, 2024.",
    "On June 10, 2023, I saw Amara Okafor present her AI project.",
    "Benjamin Lee joined the company on October 5th, 2018.",
    "Fatima Hassan gave her keynote speech on September 14, 2022.",
    "We last heard from Hiroshi Tanaka on December 31st, 2021."
]

print('| name | date | elapsed time in ms (name) | elapsed time in ms (date)')
for input in test_inputs:
    with timer() as get_time_name:
        n = extract_name(input)
    n_time = get_time_name()
    with timer() as get_time_date:
        d = extract_date(input)
    d_time = get_time_date()
    print(f'| {n} | {d} | {n_time:.2f} | {d_time:.2f} |')