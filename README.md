# Apparel Recommendation Chatbot

Welcome! This application helps you find apparel recommendations based on the "vibe" you're looking for. Describe your desired style, and the chatbot will suggest items for you.

## Getting Started: Running Locally

These instructions will guide you through setting up and running the application on your own computer. You'll need Python 3.x installed. All commands should be run from the project's root directory in your
terminal.

1. **Prepare Data Files:**
   The application needs some data files to work.

   * Ensure these two files are now located at `backend/data/Apparels_shared.csv` and `backend/data/vibe_to_attribute_examples.txt`.
2. **Set up Application Environment & Dependencies:**

   * Create a Python virtual environment. This helps manage project-specific packages:

     ```bash
     python3 -m venv backend/venv
     ```
   * Activate the virtual environment:

     * On macOS/Linux:
       ```bash
       source backend/venv/bin/activate
       ```
     * On Windows:
       ```bash
       backend\\venv\\Scripts\\activate
       ```

     (Your terminal prompt might change to show the `venv` is active).
   * Install the necessary Python packages:

     ```bash
     pip install -r backend/requirements.txt
     ```
   * Configure your API Key:

     * Open the `backend/.env` file in a text editor.
     * Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual Google Gemini API Key. The line should look like:
     * `GEMINI_MODEL_NAME_VIBE="gemini-2.5-flash-preview-05-20" `
     * `GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY"`
     * Save and close the `backend/.env` file.
3. **Run the Application:**

   * Make sure your virtual environment (from step 2) is still active.
   * Start the application server:
     ```bash
     python backend/run.py
     ```
   * You'll see some messages as the application loads. This might take a moment, especially the first time. Wait for a message indicating the server is running (e.g., on `http://127.0.0.1:8080/`).
4. **Using the Chatbot:**

   * Open your web browser (like Chrome, Firefox, or Safari).
   * Go to the address: `http://127.0.0.1:8080/`
   * The chat interface will load. You can now type in your desired vibe (e.g., "summer beach party," "cozy winter evening") and chat with the recommendation agent.
5. **Stopping the Application:**

   * To stop the server, go back to the terminal window where it's running and press `Ctrl+C`.
   * When you're done, you can deactivate the virtual environment by typing:
     ```bash
     deactivate
     ```

Enjoy finding your next favorite outfit!
