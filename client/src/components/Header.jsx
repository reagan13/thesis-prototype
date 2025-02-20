import { useLocation } from "react-router-dom"; // Import useLocation hook
import { useHeaderContext } from "../context/HeaderContext";
import TemporaryDrawer from "./TemporaryDrawer";
import Navbar from "./Navbar"; // Import Navbar

const Header = () => {
	const location = useLocation(); // Get the current route

	// Define the paths where the Navbar should be shown
	const navbarPaths = ["/result", "/INTENT", "/CATEGORY", "/NER", "/OVERALL"];

	return (
		<aside>
			<TemporaryDrawer className="" />
			{/* Conditionally render Navbar if the current path is in the navbarPaths array */}
			{/* {navbarPaths.some((path) => location.pathname.startsWith(path)) && (
				<Navbar />
			)} */}
		</aside>
	);
};

export default Header;
