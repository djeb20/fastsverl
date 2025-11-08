import os
import glob # Used to find image files

# Path to the README.md file
OUTPUT_FILENAME = "../../README.md"

# 1. Path to images RELATIVE TO THE README.md FILE
# This is used to build the <img> src tag
HTML_IMG_BASE_PATH = "shapleys/plots/mastermind_explanations" 

# 2. Path to images RELATIVE TO THIS SCRIPT
# This is used for os.path.exists() and glob.glob() checks
# It's calculated automatically
README_DIR = os.path.dirname(OUTPUT_FILENAME) # Gets "../../"
SCRIPT_IMG_BASE_PATH = os.path.join(README_DIR, HTML_IMG_BASE_PATH) # Gets "../../shapleys/plots/mastermind_explanations"

# Extension of your image files.
IMG_EXTENSION = ".png" 

# Maps the explanation names in DOMAIN_CONFIG to your image sub-directory names
IMG_DIR_MAP = {
    "Behaviour": "behaviour",
    "Outcome": "offpolicy_outcome",
    "Outcome (On-Policy)": "onpolicy_outcome",
    "Prediction": "prediction"
}

# Simplified domain configuration
# The script will auto-detect the number of states for each domain
# by counting images in the "Behaviour" sub-directory.
DOMAIN_CONFIG = {
    "Mastermind-443": {
        "explanation_names": ["Behaviour", "Outcome", "Prediction"]
    },
    "Mastermind-453": {
        "explanation_names": ["Behaviour", "Outcome", "Prediction"]
    },
    "Mastermind-463": {
        "explanation_names": ["Behaviour", "Outcome", "Prediction"]
    },
}

# --- 2. MAIN SCRIPT LOGIC ---

intro_text = """
# FastSVERL Explanations for Mastermind

This document presents qualitative examples of Shapley value explanations generated using FastSVERL for trained DQN agents in various Mastermind domains.

Mastermind is a code-breaking game where, in these versions, an agent must guess a hidden 4-letter code, drawn from a 3-letter alphabet, within a limited number of turns. After each guess, the agent receives two clues: *Clue 2* for the number of correct letters in the correct position, and *Clue 1* for the number of correct letters in the wrong position. Full details of gameplay are provided in the paper.

## How to Read the Visualisations

For each state, three explanation types are shown side-by-side: *Behaviour*, *Outcome*, and *Prediction*. The colour of each cell on the board represents its relative Shapley value, which indicates the feature's contribution to the explanation type. **For clarity, the Shapley values in all figures are normalised to sit between -1 and 1.**

* *Blue* cells indicate a *positive* contribution.
* *Red* cells indicate a *negative* contribution.
* The *intensity* of the colour corresponds to the magnitude of the influence.
* For the *Behaviour* explanation, the agent's next chosen action is marked in *green* for reference.

Full details of the FastSVERL methodology are available in the main paper.

### A Note on On-Policy Explanations

Figures for the *on-policy outcome explanations* are not presented in this README, though they are stored in the image folders. We have omitted them because the on-policy explanations were found to be erratic, suggesting potential instability as the domains scale. Investigating this is an important avenue for future work.

## Navigating This Document

This document is organised by domain size. A table of contents at the top provides direct links to each domain section. Within each section, a grid of links allows you to jump to a specific state.

Please note that the state indices are for navigational purposes only and do not represent a sequential trajectory. The states presented are representative of those encountered by an approximately optimal policy in each domain.
"""

def main():
    """Main function to generate the Markdown file."""
    md_content = [intro_text]

    # --- Create Header and Top-Level ToC ---
    md_content.append("\n## Experimental Domains")
    for domain_name in DOMAIN_CONFIG:
        anchor = domain_name.lower().replace(" ", "-")
        md_content.append(f"* [{domain_name}](#{anchor})")

    # --- Loop Through Each Domain ---
    for domain_name, config in DOMAIN_CONFIG.items():
        print(f"Processing domain: {domain_name}...")
        domain_anchor = domain_name.lower().replace(" ", "-")
        md_content.append(f"\n<br>\n\n---\n## {domain_name}")

        # Get the short domain name (e.g., "443" from "Mastermind-443")
        domain_short = domain_name.split('-')[-1]

        # --- Auto-detect number of states by counting image files ---
        # We assume the 'Behaviour' directory exists and has the images.
        behaviour_img_dir = IMG_DIR_MAP["Behaviour"]
        
        # USE THE SCRIPT-RELATIVE PATH for file system checks
        domain_img_path = os.path.join(SCRIPT_IMG_BASE_PATH, behaviour_img_dir, domain_short)
        
        # Find all images matching the pattern (e.g., "mastermind_state-*.png")
        try:
            image_files = glob.glob(os.path.join(domain_img_path, f"mastermind_state-*{IMG_EXTENSION}"))
            num_states = len(image_files)
            
            if num_states == 0:
                print(f"   - WARNING: No images found in {domain_img_path} with extension {IMG_EXTENSION}.")
                print(f"   - Skipping domain {domain_name}.")
                continue
        except FileNotFoundError: # This might not be strictly necessary with glob, but good practice
            print(f"   - ERROR: Directory not found: {domain_img_path}")
            print(f"   - Skipping domain {domain_name}.")
            continue
        
        print(f"   - Found {num_states} states.")

        # --- Create State-Level ToC Grid ---
        md_content.append("\n### Jump to State")
        links = [f"[{i + 1}](#state-{i + 1}-{domain_anchor})" for i in range(num_states)]
        num_columns = 10

        # Create the table header
        header = "| " + " | ".join([" "] * num_columns) + " |"
        separator = "|-" + "-|-".join(["-"] * num_columns) + "-|"
        md_content.append(header)
        md_content.append(separator)

        # Create the table rows
        current_row = []
        for i, link in enumerate(links):
            current_row.append(link)
            if (i + 1) % num_columns == 0:
                md_content.append("| " + " | ".join(current_row) + " |")
                current_row = []

        # Add the last row if it's not full
        if current_row:
            while len(current_row) < num_columns:
                current_row.append(" ")
            md_content.append("| " + " | ".join(current_row) + " |")
        

        # --- Loop Through Each State in the Domain ---
        for i in range(num_states):
            state_anchor = f"state-{i + 1}-{domain_anchor}"
            md_content.append(f"\n---\n<h3 id='{state_anchor}'>State {i + 1}</h3>")

            # --- Generate the side-by-side table ---
            md_content.append("<table>")

            # # Header row
            # md_content.append("<tr>")
            # for name in config['explanation_names']:
            #     md_content.append(f"<th>{name}</th>")
            # md_content.append("</tr>")
            
            # Content row (images)
            md_content.append("<tr>")
            for name in config['explanation_names']:
                explanation_dir = IMG_DIR_MAP.get(name)
                
                if not explanation_dir:
                    print(f"   - WARNING: No image directory mapping for '{name}' in state {i+1}.")
                    cell_content = f"Missing mapping for {name}"
                else:
                    # USE SCRIPT-RELATIVE PATH for os.path.exists
                    script_image_path = os.path.join(SCRIPT_IMG_BASE_PATH, explanation_dir, domain_short, f"mastermind_state-{i}{IMG_EXTENSION}")
                    
                    # USE HTML-RELATIVE PATH for the <img> src tag
                    html_image_path = f"{HTML_IMG_BASE_PATH}/{explanation_dir}/{domain_short}/mastermind_state-{i}{IMG_EXTENSION}"
                    
                    # Check if the specific file exists (using the script path)
                    if not os.path.exists(script_image_path):
                        print(f"   - WARNING: Image file not found: {script_image_path}")
                        cell_content = f"Image not found: {script_image_path}"
                    else:
                        # Create the HTML image tag (using the HTML path)
                        # We add "./" to ensure it's always treated as a relative path
                        cell_content = f'<img src="./{html_image_path}" alt="{name} explanation for state {i+1}">'
                
                md_content.append(f'<td valign="top" align="center">{cell_content}</td>')
            
            md_content.append("</tr>")
            md_content.append("</table>")
    
    # --- Write to file ---
    with open(OUTPUT_FILENAME, "w") as f:
        f.write("\n".join(md_content))
    
    print(f"\nSuccessfully generated {OUTPUT_FILENAME}!")


if __name__ == "__main__":
    # Ensure the script uses paths relative to its own location
    # This makes os.path.join work correctly
    try:
        script_dir = os.path.dirname(__file__)
        if script_dir:
            os.chdir(script_dir)
    except NameError:
        # __file__ is not defined, e.g., in an interactive REPL
        # In this case, we assume the script is run from the correct directory
        pass 
        
    main()