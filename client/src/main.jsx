// index.js
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import Chat from "./pages/Chat";
import LatestChatLoader from "./utils/LatestChatLoader"; // Import the LatestChatLoader
import ChatAnalysis from "./pages/ChatAnalysis";

// index.js
import NewChatLoader from "./utils/NewChatLoader"; // Import the NewChatLoader

const router = createBrowserRouter([
	{
		path: "/",
		element: <App />,
		children: [
			{
				index: true, // Redirect to the latest chat when accessing the root path "/"
				element: <LatestChatLoader />,
			},
			{
				path: "chat/:id", // Render the Chat component for a specific chat
				element: <Chat />,
			},
			{
				path: "chat", // Redirect to the latest chat if no ID is provided
				element: <LatestChatLoader />,
			},
			{
				path: "chat/new", // Create a new chat
				element: <NewChatLoader />,
			},
			{
				path: "chat-analysis/:id", // Route for Chat Analysis page
				element: <ChatAnalysis />,
			},
		],
	},
]);

ReactDOM.createRoot(document.getElementById("root")).render(
	<React.StrictMode>
		<RouterProvider router={router} />
	</React.StrictMode>
);
