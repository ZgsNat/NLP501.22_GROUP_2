import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_agent.criteria.jd_parser import parse_jd  # noqa: E402
from cv_agent.ingest.pipeline import ingest_text  # noqa: E402
from cv_agent.schema import CriteriaProfile  # noqa: E402

SENIOR_CV = """Nguyen Van Hung
Senior Automation Test Engineer
Email: hung.nv@example.com
Phone: 0912 345 678
Date of birth: 12/05/1992
Gender: Male
Address: 12 Le Loi Street, District 1

SUMMARY
6 years of experience in software testing.
EXPERIENCE
Company A (2019 - Present)
- Designed and built a Page Object Model automation framework with Selenium WebDriver and Java.
- Integrated the suite into the Jenkins pipeline; API testing with Rest Assured and Postman.
SKILLS
Programming: Java, Python
Certification: ISTQB CTFL
Languages: English (TOEIC 800)
"""

MANUAL_CV = """Tran Thi Lan
QA Engineer
Email: lan@example.com
SUMMARY
1 years of experience in software testing.
- Executed manual test cases and reported defects in Jira.
SKILLS
Languages: English
"""

FPT_LIKE_JD = """Automation Test Engineer
Requirements
- At least 2 years of experience in automation testing
- Hands-on with Selenium WebDriver, Playwright or Appium
- Programming skills in Java or Python
Nice to have
- ISTQB Foundation certificate is a plus
- Experience with Jenkins or GitLab CI
"""


@pytest.fixture
def senior_cv():
    return ingest_text(SENIOR_CV, "senior")


@pytest.fixture
def manual_cv():
    return ingest_text(MANUAL_CV, "manual")


@pytest.fixture
def jd():
    return parse_jd(FPT_LIKE_JD, "jd_test")


@pytest.fixture
def profile():
    return CriteriaProfile("Test", {"tools": 0.25, "depth": 0.2, "languages": 0.15, "test_types": 0.1, "cicd": 0.1,
                                    "experience": 0.1, "certification": 0.05, "foreign_lang": 0.05})
