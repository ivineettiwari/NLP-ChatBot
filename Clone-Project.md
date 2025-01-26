Here are the detailed steps to clone the project and run it using a virtual environment (`venv`) on Windows:

### 1. Install Git (if not already installed)
First, ensure you have Git installed on your Windows machine. You can download it from [here](https://git-scm.com/downloads).

After downloading, follow the installation steps, making sure to check the option to add Git to your system's PATH.

### 2. Clone the Project
Open Command Prompt or PowerShell and navigate to the directory where you want to clone the project. Run the following command:

```bash
git clone https://github.com/ivineettiwari/NLP-ChatBot.git
```

Replace `<repository_url>` with the actual URL of the GitHub repository (e.g., `https://github.com/yourusername/chatbot.git`).

Once the repository is cloned, navigate into the project directory:

```bash
cd `<folder_name>`
```

### 3. Install Python (if not already installed)
Make sure Python 3.x is installed on your system. You can download Python from the official website: [Python Downloads](https://www.python.org/downloads/). During installation, make sure to check the option to "Add Python to PATH."

### 4. Create a Virtual Environment
Now, create a virtual environment in your project directory to isolate the dependencies for the project:

```bash
python -m venv venv
```

This will create a `venv` directory within your project folder, which will contain a fresh Python environment.

### 5. Activate the Virtual Environment
To activate the virtual environment, run the following command:

- **For Command Prompt**:

  ```bash
  venv\Scripts\activate
  ```

- **For PowerShell**:

  ```bash
  .\venv\Scripts\Activate.ps1
  ```

You should see `(venv)` appear in the command line, indicating that the virtual environment is active.

### 6. Install Dependencies
With the virtual environment activated, you can now install the required project dependencies. If you have a `requirements.txt` file in the project directory, install the dependencies using the following command:

```bash
pip install -r requirements_new.txt
```


### 7. Train the Model
Now, you can train the chatbot model using the `chatbot.py` script. Run the script as follows:

```bash
python jobs.py
```

This will train the chatbot model and save it as `chatbot_model.tflearn` in the `model/` directory.

### 8. Run the Flask App (Optional)
If you want to integrate the chatbot into a web interface, run the Flask app by executing:

```bash
python app.py
```

This will start a Flask web server. By default, it runs on `http://127.0.0.1:8000/`, and you can interact with the chatbot via POST requests to `/get`.

### 9. Deactivate the Virtual Environment
After you're done, you can deactivate the virtual environment by running:

```bash
deactivate
```

This will return you to the global Python environment.

### 10. Additional Notes
- You can always reinstall the dependencies by running `pip install -r requirements_new.txt` if needed.
- To update the project, simply pull the latest changes using `git pull` in the project directory.
- If the project has any specific instructions for additional configurations, make sure to follow those as well.

---

Let me know if you need further assistance with any of the steps!