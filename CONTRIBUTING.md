# **Contributing to `phicloudmask`** 🤝

We welcome contributions from the community! Every contribution, no matter how small, is appreciated and credited. Here’s how you can get involved:

## **How to contribute** 🛠️

1. **Fork the repository:** Start by forking the [phicloudmask](https://github.com/IPL-UV/phicloudmask) repository to your GitHub account. 🍴

2. **Clone your fork locally:**

    ```bash
    cd <directory_in_which_repo_should_be_created>
    git clone https://github.com/IPL-UV/phicloudmask.git
    cd phicloudmask
    ```

3. **Set up your local environment:** 🌱

   - If you're using `pyenv`, select a Python version:
     ```bash
     pyenv local <x.y.z>
     ```
   - Install dependencies and activate the environment:
     ```bash
     poetry install
     poetry shell
     ```
   - Install pre-commit hooks:
     ```bash
     poetry run pre-commit install
     ```

4. **Create a branch for your changes:** 🖋️

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

5. **Make your changes:** Develop your feature or fix, ensuring to write clear, concise commit messages and include any necessary tests.

6. **Run checks on your changes:** ✅

   - Run formatting checks:
     ```bash
     make check
     ```
   - Run unit tests:
     ```bash
     make test
     ```
   - Optionally, run tests across different Python versions using tox:
     ```bash
     tox
     ```

7. **Commit your changes and push your branch:** 🚀

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

8. **Submit a pull request:** Go to your repository on GitHub and create a pull request to the `main` branch of the `phicloudmask` repository. Make sure your pull request meets the following guidelines:

   - Include tests for your changes.
   - Update documentation if your pull request adds functionality.
   - Provide a detailed description of your changes.

## **Types of contributions** 📦

- **Report Bugs:** 🐛 Report bugs by creating an issue on the [phicloudmask GitHub repository](https://github.com/IPL-UV/phicloudmask/issues). Please include:
  - Your operating system name and version.
  - Details about your local setup that might be helpful in troubleshooting.
  - Detailed steps to reproduce the bug.

- **Fix Bugs:** 🛠️ Look through the GitHub issues for bugs tagged with "bug" and "help wanted". These are open for anyone who wants to contribute a fix.

- **Implement Features:** ✨ Help implement new features by checking issues tagged with "enhancement" and "help wanted".

- **Write Documentation:** 📚 `phicloudmask` can always benefit from improved documentation. You can contribute by enhancing the official documentation, writing clear docstrings, or even creating blog posts and tutorials.

- **Submit Feedback:** 💬 Propose new features or provide feedback by filing an issue on the [phicloudmask GitHub repository](https://github.com/IPL-UV/phicloudmask/issues).
  - If you propose a new feature, please explain in detail how it would work and keep the scope narrow to make implementation easier.
  - Remember that this is a community-driven project, and every bit of feedback is valuable!

## **Get Started!** 🚀

Ready to contribute? Follow the steps above to set up `phicloudmask` for local development and start making your mark on the project. We’re excited to see what you’ll contribute!