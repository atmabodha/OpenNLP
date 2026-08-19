import os
import subprocess
import sys

from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv("api_key.env")
load_dotenv(env_path)

GRAFT_CONFIG = {
    "GRAFT_PROVIDER": "openai",
    "GRAFT_BASE_URL": "https://openrouter.ai/api/v1",
    "GRAFT_API_KEY": os.environ["OPENROUTER_API_KEY"],
    "GRAFT_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free"
}

def verify_graft_installed():
    """Checks if the Graft CLI is available in the system PATH."""
    try:
        subprocess.run(
            ["graft", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Graft CLI is not installed or not in your PATH.")
        print("Run 'npm install -g @nanonets/graft' to install it.")
        sys.exit(1)

def run_deep_build():
    """Executes the deep build process with the injected environment variables."""
    # Copy current system environment variables so we don't break existing paths
    env = os.environ.copy()
    
    # Inject our OpenRouter config into the localized subprocess environment
    for key, value in GRAFT_CONFIG.items():
        env[key] = value

    print(f"🚀 Starting Graft deep build using model: {GRAFT_CONFIG['GRAFT_MODEL']}...")
    
    try:
        # Run the command and stream the output to the console in real-time
        process = subprocess.Popen(
            ["graft", "build", "--deep"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Line buffered
        )

        # Stream output line-by-line
        for line in process.stdout:
            print(line, end="")
            
        process.wait()

        if process.returncode == 0:
            print("\n✅ Build complete! Your graph is ready in the 'graft/' directory.")
        else:
            print(f"\n❌ Build failed with exit code {process.returncode}.")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\n⚠️ Build interrupted by user. Shutting down gracefully...")
        process.terminate()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 1. Ensure the API key has been changed from the default template
    if "your-actual-api-key-here" in GRAFT_CONFIG["GRAFT_API_KEY"]:
        print("❌ Error: Please insert your OpenRouter API key into the script.")
        sys.exit(1)

    # 2. Verify installation
    verify_graft_installed()
    
    # 3. Execute
    run_deep_build()

# Once this is done, run: graft viz

## CLAUDE CODE SETUP AND RUN COMMAND
# .env
"""
ANTHROPIC_BASE_URL=https://openrouter.ai/api
ANTHROPIC_AUTH_TOKEN=sk-or-v1-asdaxd2e1c
ANTHROPIC_API_KEY=""
ANTHROPIC_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
"""
# set -a && source .env && set +a && claude

## CLAUDE CODE SAMPLE PROMPTS
"""
1.Prompt 1: Isolate a Leaf Node:
Paste this into Claude to have it analyze the Graft map and pick a safe starting point:
######################
Read the codebase map in the graft/ directory and analyze graft/.graph/wiring.json. 
Find a single 'leaf node' function or class for us to start refactoring. 
I want something that is relatively isolated—meaning it might be called by other files, 
but it does not call out to many external dependencies itself. 
Tell me which function you chose, what it does based on the AI concept notes, 
and why it is a safe starting point.
######################

2.Prompt 2: Write Characterization Tests:
Once Claude has picked a function (and you agree with the choice), paste this to lock down the existing behavior without changing the code:
######################
Let's focus on the function you just identified. 
Before we change any code, write a comprehensive suite of characterization tests for it. 
Do not test what the function should do — test exactly what it currently does, 
including any edge cases or error states mentioned in the Graft concept files. 
Save the test file, then run the test suite to prove that all tests currently pass 
against the legacy code.
######################

3.Prompt 3: Refactor and Validate:
Only after Claude successfully runs the tests and they pass, paste this to execute the actual refactoring:
######################
The characterization tests are passing and the current behavior is locked in.
Now, refactor the target function. Improve the naming, modernize the syntax, 
and untangle the logic to make it cleaner. 
Address any specific failure points that Graft originally warned us about. 
Once you are done, run the characterization tests again. 
If they fail, immediately revert your changes, figure out why the behavior drifted, 
and try again until the tests pass.
######################

"""

## CLAUDE SKILLS
"""
mkdir -p .claude/skills/characterization
touch .claude/skills/characterization/SKILL.md
"""