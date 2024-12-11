import * as React from "react";
import * as ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import GPT2 from "./pages/GPT2";
import DistilBERT from "./pages/distilbert";
import Home from "./pages/Home";
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
				path: "gpt2",
				element: <GPT2 />,
			},
			{
				path: "distilbert/:id",
				element: <DistilBERT />,
			},
		],
	},
]);

ReactDOM.createRoot(document.getElementById("root")).render(
	<React.StrictMode>
		<RouterProvider router={router} />
	</React.StrictMode>
);
