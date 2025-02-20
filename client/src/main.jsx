import * as React from "react";
import * as ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import GPT2 from "./pages/GPT2";
import DistilBERT from "./pages/Distilbert";
import Home from "./pages/Home";
import GraphPage from "./components/Graphpage";
import Chat from "./pages/Chat";
const router = createBrowserRouter([
	{
		path: "/",
		element: <App />,
		children: [
			{
				index: true, // This makes the Home component render at the root path "/"
				element: <Home />,
			},
			{
				path: "home", // This allows the Home component to render at "/home"
				element: <Home />,
			},
			{
				path: "chat/:id",
				element: <Chat />,
			},
			{
				path: "gpt2",
				element: <GPT2 />,
			},
			{
				path: "result",
				element: <DistilBERT />,
			},
			{
				path: "result/:id",
				element: <DistilBERT />,
			},
			{
				path: "INTENT",
				element: <GraphPage />,
			},
			{
				path: "CATEGORY",
				element: <GraphPage />,
			},
			{
				path: "NER",
				element: <GraphPage />,
			},
			{
				path: "OVERALL",
				element: <GraphPage />,
			},
		],
	},
]);

ReactDOM.createRoot(document.getElementById("root")).render(
	<React.StrictMode>
		<RouterProvider router={router} />
	</React.StrictMode>
);
