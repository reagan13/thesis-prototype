import { HeaderProvider } from "./context/HeaderContext"; // Import HeaderProvider
import Header from "./components/Header";

import "./App.css";
import { Outlet } from "react-router-dom";
import { DataProvider } from "./context/DataContext";

const App = () => {
	return (
		<DataProvider>
			<HeaderProvider>
				<div className="flex h-screen">
					<Header /> {/* Header now uses context */}
					<div className="h-full w-full pt-5 pb-3 ">
						<Outlet />
					</div>
				</div>
			</HeaderProvider>
		</DataProvider>
	);
};

export default App;
