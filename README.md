This project provides a async client for the EDGAR REST API which structures the responses into pydantic schemas.

The following purposes include adding financial statement reader classes, computing common metrics such as FCF, and adding classes to spot investment red flags.

Finally, perhaps adding, as a fallback to the EDGAR REST API, the pipeline of downloading, extracting and loading into the pydantic schemas the data directly from edgar txt/xbrl submissions.
