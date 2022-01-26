import { REPO_NAME, REPO_OWNER } from './common';
// Create a personal access token at https://github.com/settings/tokens/new?scopes=repo
const octokit = new Octokit({ auth: `personal-access-token123` });

const response = await octokit.request(
  `GET /repos/${REPO_OWNER}/${REPO_NAME}/issues`,
  {
    accept: 'application/vnd.github.v3+json',
  }
);
