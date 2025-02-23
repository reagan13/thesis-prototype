import { useLocation } from "react-router-dom"; // Import useLocation hook

import TemporaryDrawer from "./TemporaryDrawer";


const Header = () => {
	const location = useLocation(); // Get the current route

	// Define the paths where the Navbar should be shown
	const navbarPaths = ["/result", "/INTENT", "/CATEGORY", "/NER", "/OVERALL"];

	return (
		<aside>
			<TemporaryDrawer className=" " />
		
		</aside>
	);
};

export default Header;
