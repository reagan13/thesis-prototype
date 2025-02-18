import { Link } from "react-router-dom";
import { useData } from "../context/DataContext"; // Import the context

const Navbar = () => {
    const { setGraphType, isSidebarCollapsed } = useData(); // Get the setGraphType function

    return (
        <nav className="flex justify-center items-center space-x-[115px] text-white text-lg"
        style={{
          paddingLeft: isSidebarCollapsed ? "100px" : "280px",
          transition: "all 0.3s ease",
        }}
      >
            <Link
                to="/INTENT"
                className="hover:underline"
                onClick={() => setGraphType("INTENT")} // Set the graph type when clicked
            >
                Intent Graph
            </Link>
            <Link
                to="/CATEGORY"
                className="hover:underline"
                onClick={() => setGraphType("CATEGORY")} // Set the graph type when clicked
            >
                Category Graph
            </Link>
            <Link
                to="/NER"
                className="hover:underline"
                onClick={() => setGraphType("NER")} // Set the graph type when clicked
            >
                NER Graph
            </Link>
            <Link
                to="/OVERALL"
                className="hover:underline"
                onClick={() => setGraphType("OVERALL")} // Set the graph type when clicked
            >
                Overall Graph
            </Link>
        </nav>
    );
};

export default Navbar;
