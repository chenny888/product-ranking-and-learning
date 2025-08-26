# GitHub Upload Instructions

Follow these steps to upload your research code to GitHub:

## Step 1: Prepare the Repository

Your local repository is already initialized and ready. You have:
- ✅ Git repository initialized
- ✅ README.md with comprehensive documentation
- ✅ pyproject.toml with dependency management
- ✅ .gitignore configured
- ✅ All necessary files present

## Step 2: Create GitHub Repository

1. **Go to GitHub**: Visit https://github.com and sign in to your account

2. **Create new repository**:
   - Click the "+" icon in the top right corner
   - Select "New repository"
   - Repository name: `revenue-management-experiments` (or your preferred name)
   - Description: "Replication package for 'Revenue Management with Ranking Learning' - numerical experiments and algorithms"
   - Choose visibility: 
     - **Public** (recommended for research reproducibility)
     - **Private** (if you want to keep it private initially)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

## Step 3: Connect Local Repository to GitHub

After creating the GitHub repository, run these commands in your terminal:

```bash
# Navigate to your project directory
cd "/Users/chenny/Library/CloudStorage/Dropbox/Research/revenue_management/2019ranking_learning/CODE/python_code_final/upload"

# Add all files to git
git add .

# Commit the files
git commit -m "Initial commit: Add revenue management experiments code and documentation"

# Add the GitHub remote (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/revenue-management-experiments.git

# Push to GitHub
git push -u origin master
```

## Step 4: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. The README.md will be displayed automatically on the main page

## Step 5: Update pyproject.toml URLs (Optional)

After uploading, update the URLs in your `pyproject.toml` file:

```bash
# Edit pyproject.toml to replace placeholder URLs with your actual GitHub repository
# Update these lines with your actual username:
# Homepage = "https://github.com/YOUR_USERNAME/revenue-management-experiments"
# Repository = "https://github.com/YOUR_USERNAME/revenue-management-experiments"
# Issues = "https://github.com/YOUR_USERNAME/revenue-management-experiments/issues"
```

Then commit the changes:
```bash
git add pyproject.toml
git commit -m "Update repository URLs in pyproject.toml"
git push
```

## Step 6: Add Repository Features (Recommended)

### Add Topics/Tags
1. Go to your repository on GitHub
2. Click the gear icon next to "About" 
3. Add topics: `revenue-management`, `optimization`, `machine-learning`, `operations-research`, `python`

### Create Release (Optional)
1. Go to "Releases" on the right sidebar
2. Click "Create a new release"
3. Tag: `v1.0.0`
4. Title: "Initial Release - Replication Package"
5. Description: Brief description of the code and experiments
6. Click "Publish release"

### Enable GitHub Pages (Optional)
If you want to host documentation:
1. Go to repository Settings
2. Scroll to "Pages" section
3. Select source: "Deploy from a branch"
4. Choose branch: "master" or "main"
5. Folder: "/ (root)"

## Troubleshooting

### Authentication Issues
If you encounter authentication issues:

1. **Use Personal Access Token** (recommended):
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with repo permissions
   - Use token instead of password when prompted

2. **Or use SSH** (alternative):
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/revenue-management-experiments.git
   ```

### Large File Issues
If you get warnings about large files:
- The pickle files might be large
- Consider using Git LFS for files > 50MB
- Or exclude them from the repository if reproducibility allows

## Next Steps After Upload

1. **Share the repository**: Include the GitHub link in your paper
2. **Add collaborators**: If working with co-authors
3. **Create issues**: For any known limitations or future improvements
4. **Add license**: Consider adding an appropriate open-source license
5. **Update citation**: Add proper citation information to README

## Repository Structure on GitHub

Your uploaded repository will have this structure:
```
revenue-management-experiments/
├── README.md                           # Main documentation
├── pyproject.toml                      # Dependencies & metadata  
├── uv.lock                            # Locked dependency versions
├── validate_setup.py                  # Environment validation script
├── utils.py                           # Core algorithms
├── Offline_Experiments_Section_6_1.ipynb  # Offline experiments
├── Online_experiment_Section_6_2.py       # Online experiments
├── online_data_iter_*.pickle              # Pre-computed results
├── ranking_OR_review_round_2.pdf          # Research paper
├── .gitignore                             # Git ignore rules
└── .python-version                        # Python version spec
```

Users will be able to:
- Clone the repository
- Install dependencies with `uv sync`
- Run validation with `uv run python validate_setup.py`
- Reproduce all experiments following the README instructions
