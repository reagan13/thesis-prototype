import React, { createContext, useContext, useState } from "react";
import useLocalStorage from "../hooks/useLocalStorage"; // Import the custom hook

const DataContext = createContext();

export const DataProvider = ({ children }) => {
    const [data, setData] = useLocalStorage("chatData", { messages: [] }); // Use local storage for chat data
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false); // Sidebar state
    const [graphType, setGraphType] = useState(""); // Add graphType state

    return (
        <DataContext.Provider value={{ data, setData, isSidebarCollapsed, setIsSidebarCollapsed, graphType, setGraphType }}>
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
