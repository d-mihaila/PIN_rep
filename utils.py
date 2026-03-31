from langchain_openai import ChatOpenAI
from langchain_core.language_models import LanguageModelInput
from typing import Any, List, Optional


# this will look for the regular openai env vars
# (OPENAI_API_KEY and OPENAI_API_BASE so override externally with gpt-jrc coords)
class JRCChatOpenAI(ChatOpenAI):

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """
        NOTE: this solves bug_00_tool_message: by changing role of tool messages to system
            gpt@jrc is happy
        """

        r = super()._get_request_payload(input_=input_, stop=stop, **kwargs)
        for m in r["messages"]:
            if m["role"] == "tool":
                m["role"] = "system"
        return r