# GitHub API functions
def get_github_stars_forks(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        data = response.json()
        return data.get('stargazers_count', 0), data.get('forks_count', 0)
    else:
        print(f"Error fetching stars and forks: {response.status_code}")
        return 0, 0

def get_github_issues(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        issues = response.json()
        return len(issues)
    else:
        print(f"Error fetching issues: {response.status_code}")
        return 0

def get_github_pull_requests(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        pulls = response.json()
        return len(pulls)
    else:
        print(f"Error fetching pull requests: {response.status_code}")
        return 0

def get_github_license(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        data = response.json()
        return data['license']['name'] if data.get('license') else 'No license found'
    else:
        print(f"Error fetching license: {response.status_code}")
        return 'No license found'

def get_last_commit(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        commits = response.json()
        return commits[0]['commit']['committer']['date'] if commits else 'No commits found'
    else:
        print(f"Error fetching commits: {response.status_code}")
        return 'No commits found'

def get_github_workflow_status(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 200:
        runs = response.json()
        return runs['workflow_runs'][0]['status'] if runs.get('workflow_runs') else "No workflows found"
    else:
        print(f"Error fetching workflow status: {response.status_code}")
        return 'No workflows found'

# Main function to display OSINT data
def fetch_osint_data(owner, repo):
    stars, forks = get_github_stars_forks(owner, repo)
    open_issues = get_github_issues(owner, repo)
    open_pulls = get_github_pull_requests(owner, repo)
    license_type = get_github_license(owner, repo)
    last_commit = get_last_commit(owner, repo)
    workflow_status = get_github_workflow_status(owner, repo)

    # Print the collected data
    print(f"GitHub Repository: {owner}/{repo}")
    print(f"Stars: {stars}, Forks: {forks}")
    print(f"Open Issues: {open_issues}, Open Pull Requests: {open_pulls}")
    print(f"License: {license_type}")
    print(f"Last Commit: {last_commit}")
    print(f"Workflow Status: {workflow_status}")

# Example usage
if __name__ == "__main__":
    fetch_osint_data("Chemically-Motivated-Solutions", "OSINT_Tool")  # Replace with your desired repo