import React, { createContext, useContext, useState } from "react";
import useLocalStorage from "../hooks/useLocalStorage";

const DataContext = createContext();

export const DataProvider = ({ children }) => {
    // Use local storage for chat data
    const [data, setData] = useLocalStorage("chatData", {
        chatHistory: [],
        activeChatId: null,
        selectedModel: "Baseline", // Default to Baseline Model
    });
    const [selectedChatAnalysis, setSelectedChatAnalysis] = useState(null);

    // Function to update the selected model
    const setSelectedModel = (model) => {
        setData((prevData) => ({
            ...prevData,
            selectedModel: model,
        }));
    };

    // Ensure chat history is sorted by creation time (newest first)
    const sortedChatHistory = Array.isArray(data.chatHistory) ? [...data.chatHistory].sort((a, b) => b.id - a.id) : [];

    // Set the active chat to the latest chat if none is selected
    const activeChat =
        data.activeChatId !== null
            ? sortedChatHistory.find((chat) => chat.id === data.activeChatId)
            : sortedChatHistory[0]; // Default to the latest chat

    const createNewChat = () => {
        const newChatId = Date.now();
        const newChat = {
            id: newChatId,
            title: `Conversation ${newChatId}`,
            messages: [],
        };
        setData((prevData) => ({
            ...prevData,
            chatHistory: [newChat, ...prevData.chatHistory], // Add new chat to the top
            activeChatId: newChatId, // Set the new chat as active
        }));
    };

    // Delete a chat
    const deleteChat = (id) => {
        setData((prevData) => {
            const updatedChatHistory = prevData.chatHistory.filter(
                (chat) => chat.id !== id
            );

            const newActiveChatId =
                prevData.activeChatId === id && updatedChatHistory.length > 0
                    ? updatedChatHistory[0].id // Set the first chat as active
                    : null;

            return {
                ...prevData,
                chatHistory: updatedChatHistory,
                activeChatId: newActiveChatId,
            };
        });
    };

    const switchChat = (id) => {
        setData((prevData) => {
            if (!prevData.chatHistory.some((chat) => chat.id === id)) {
                console.warn(`Chat with ID ${id} not found.`);
                return prevData; // Do not update if the chat does not exist
            }
            return {
                ...prevData,
                activeChatId: id,
            };
        });
    };

    // Add a message to the active chat
    const addMessage = (message) => {
        setData((prevData) => {
            const updatedChatHistory = prevData.chatHistory.map((chat) =>
                chat.id === prevData.activeChatId
                    ? {
                            ...chat,
                            messages: [
                                ...chat.messages,
                                { ...message, id: Date.now() }, // Ensure a unique ID is assigned
                            ],
                      }
                    : chat
            );
            return {
                ...prevData,
                chatHistory: updatedChatHistory,
            };
        });
    };

    const [selectedBotResponse, setSelectedBotResponse] = useState(null);

    return (
        <DataContext.Provider
            value={{
                data: {
                    ...data,
                    chatHistory: sortedChatHistory, // Always provide sorted chat history
                    activeChatId: activeChat?.id || null, // Ensure activeChatId is valid
                },
                setData,
                setSelectedModel,
                createNewChat,
                deleteChat,
                switchChat,
                addMessage,
                selectedBotResponse,
                setSelectedBotResponse, // Ensure this is included
                selectedChatAnalysis,
                setSelectedChatAnalysis,
            }}
        >
            {children}
        </DataContext.Provider>
    );
};

export const useData = () => {
    const context = useContext(DataContext);
    if (!context) {
        throw new Error("useData must be used within a DataProvider");
    }
    return context;
};