import { useState } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import ModelDropdown from "./dropdown";
import { useData } from "../context/DataContext";
import axios from "axios";

const ChatInterface = () => {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const { data, addMessage } = useData();

  const activeChat = data.chatHistory.find(
    (chat) => chat.id === data.activeChatId
  );

  const handleSend = async () => {
    if (!input.trim()) return;

    setLoading(true);

    try {
      const apiUrlMap = {
        Baseline: "http://127.0.0.1:5000/baseline",
        Concat: "http://127.0.0.1:5000/concat",
        Crossattention: "http://127.0.0.1:5000/crossattention",
        Dense: "http://127.0.0.1:5000/dense",
      };

      const selectedModel = data.selectedModel || "Baseline";
      const payload = { text: input, session_id: "default" };
      console.log("Sending request:", {
        model: selectedModel,
        url: apiUrlMap[selectedModel],
        payload,
      });
      let baselineResponse, hybridResponse;

      if (selectedModel === "Baseline") {
        baselineResponse = await axios.post(apiUrlMap["Baseline"], {
          text: input,
          session_id: "default",
        });
      } else {
        [baselineResponse, hybridResponse] = await Promise.all([
          axios.post(apiUrlMap["Baseline"], {
            text: input,
            session_id: "default",
          }),
          axios.post(apiUrlMap[selectedModel], {
            text: input,
            session_id: "default",
          }),
        ]);
      }

      const primaryResponse =
        selectedModel === "Baseline" ? baselineResponse : hybridResponse;

      const predictions = {
        baseline: baselineResponse.data,
      };
      if (selectedModel !== "Baseline") {
        predictions[selectedModel.toLowerCase()] = primaryResponse.data;
      }
      console.log("Predictions:", predictions);
      const userMessage = {
        text: input,
        sender: "user",
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        botResponse: {
          text: primaryResponse.data.response, // Maps to "response"
          timestamp: new Date().toLocaleString(),
          category: {
            label: primaryResponse.data.classification.category.label,
            confidence: `${(
              primaryResponse.data.classification.category.confidence * 100
            ).toFixed(2)}%`,
          },
          intent: {
            label: primaryResponse.data.classification.intent.label,
            confidence: `${(
              primaryResponse.data.classification.intent.confidence * 100
            ).toFixed(2)}%`,
          },
          ner: primaryResponse.data.classification.ner.map((entity) => ({
            entity: entity.entity,
            label: entity.label,
            confidence: `${(entity.confidence * 100).toFixed(2)}%`,
          })),
          classified_input: primaryResponse.data.classified_input, // CHANGE: Use backend's classified_input
          metrics: {
            classification_time:
              primaryResponse.data.metrics.classification_time.toFixed(2),
            generation_time:
              primaryResponse.data.metrics.generation_time.toFixed(2),
            memory_usage: primaryResponse.data.metrics.memory_usage.toFixed(2),
            overall_time: primaryResponse.data.metrics.overall_time.toFixed(2),
          },
          modelUsed: selectedModel,
          id: Date.now() + 1,
          predictions: predictions,
        },
      };

      addMessage(userMessage);
      setInput("");
    } catch (error) {
      console.error("Error:", error);
      const errorMsg = error.response
        ? `${error.message}: ${JSON.stringify(error.response.data)}`
        : error.message;
      console.log("Error details:", error.response?.data); // Log backend error message
      alert(`An error occurred: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full h-full max-h-screen">
      {loading && <CircularProgress className="absolute top-1/2 left-1/2" />}
      <div className="h-full px-36">
        <div className="flex flex-col h-full gap-5">
          <div className="flex justify-end z-index-20">
            <ModelDropdown />
          </div>
          <div className="flex-1 overflow-y-auto scrollbar-hide">
            <MessageDisplay />
          </div>
          <InputSection
            input={input}
            setInput={setInput}
            handleSend={handleSend}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
