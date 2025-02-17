import { useState } from "react";
import axios from "axios";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import { useData } from "../context/DataContext";

const ChatInterface = () => {
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const { data, setData, isSidebarCollapsed } = useData(); // Get sidebar state
    const messages = data.messages || [];

    const handleSend = async () => {
        if (!input.trim()) return;
        const inputData = { text: input };
        setLoading(true);

        try {
            const [categoryResponse] = await Promise.all([
                axios.post("http://localhost:5000/baseline_category", inputData),
            ]);

            const botResponse = {
                text: `The predicted category is "${categoryResponse.data.class}".`,
                sender: "bot",
                id: Date.now() + 1,
            };

            const userMessage = {
                text: input,
                sender: "user",
                id: Date.now(),
                botResponse,
            };

            setData({ messages: [...messages, userMessage] });
            setInput("");
        } catch (error) {
            console.error("Error:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full h-full flex items-center justify-center bg-[#0e0e0f]">
            {loading && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50">
                    <CircularProgress size={150} />
                </div>
            )}
            <div
                className={`p-6 bg-[#133075d2] outline outline-2 outline-white outline-offset-2 rounded-lg h-[640px] mb-[20px]`}
					style={{
						width: isSidebarCollapsed ? "1350px" : "1200px",
						marginLeft: isSidebarCollapsed ? "100px" : "300px",
						transition: "all 0.3s ease",
                       
					}}
            >
                <div className={"space-y-6 text-center justify-between flex flex-col h-full"} >
                    <MessageDisplay />
                    <InputSection input={input} setInput={setInput} handleSend={handleSend} />
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
