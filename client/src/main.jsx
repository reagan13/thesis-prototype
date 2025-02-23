import * as React from "react";
import * as ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";

import Home from "./pages/Home";

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
			
			
		],
	},
]);

ReactDOM.createRoot(document.getElementById("root")).render(
	<React.StrictMode>
		<RouterProvider router={router} />
	</React.StrictMode>
);
