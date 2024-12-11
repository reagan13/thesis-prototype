import React, { createContext, useContext, useEffect } from "react";
import useLocalStorage from "../hooks/useLocalStorage"; // Import the custom hook

const DataContext = createContext();

export const DataProvider = ({ children }) => {
	const [data, setData] = useLocalStorage("chatData", { messages: [] }); // Use local storage for chat data

	// Load data from local storage on mount
	useEffect(() => {
		const storedData = localStorage.getItem("chatData");
		if (storedData) {
			try {
				const parsedData = JSON.parse(storedData);
				// Only set data if it's different from the current state
				if (JSON.stringify(parsedData) !== JSON.stringify(data)) {
					setData(parsedData); // Set the entire data object
				}
			} catch (error) {
				console.error("Error parsing stored data:", error);
			}
		}
	}, [setData, data]); // Add data to dependencies to prevent infinite loop

	// Save data to local storage whenever it changes
	useEffect(() => {
		localStorage.setItem("chatData", JSON.stringify(data));
	}, [data]); // This is correct

	return (
		<DataContext.Provider value={{ data, setData }}>
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
