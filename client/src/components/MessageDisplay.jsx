import { useEffect } from "react";
import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import PropTypes from "prop-types";
import { useData } from "../context/DataContext"; // Import useData

const MessageDisplay = () => {
  const { data } = useData(); // Get data from context
  const messages = data.messages || []; // Access messages from context

  return (
    <div className={"h-full overflow-y-auto max-h-[540px] border border-gray-300 rounded-lg p-10 space-y-5 bg-[#0A0F24] text-white"} style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}>
      {messages.map((message) => {
        if (!message || !message.botResponse) {
          console.error("Invalid message format:", message);
          return null; // Skip rendering if message is invalid
        }

        return (
          <div key={message.id} className="space-y-10">
            <UserMessage text={message.text} />

            <BotMessage
              text={message.botResponse.text} // Accessing bot response text
              category={message.botResponse.categoryResponse?.data?.class || "Unknown"} // Fallback to "Unknown" if data is missing
              intent={message.botResponse.intentResponse?.data?.class || "Unknown"} // Fallback to "Unknown" if data is missing
              ner={"Unknown"}
              id={message.botResponse.id}
            />
          </div>
        );
      })}
    </div>
  );
};

export default MessageDisplay;