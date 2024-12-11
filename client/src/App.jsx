import { HeaderProvider } from "./context/HeaderContext"; // Import HeaderProvider
import Header from "./components/Header";
import Footer from "./components/Footer";
import "./App.css";
import { Outlet } from "react-router-dom";
import { DataProvider } from "./context/DataContext";

const App = () => {
	return (
		<DataProvider>
			<HeaderProvider>
				<div className="flex flex-col h-screen bg-white">
					<Header /> {/* Header now uses context */}
					<div className="h-full w-full">
						<Outlet />
					</div>
				</div>
			</HeaderProvider>
		</DataProvider>
	);
};

export default App;
