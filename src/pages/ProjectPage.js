import React, { Component } from 'react';
import { MainLayout } from '../components/MainLayout';
import { ProjectSummary, AwardsComponent, SlideshowComponent, TableComponent, ListComponent } from '../components/PageComponents';
import styles from '../SCSS/Main.module.scss';

const componentsList = {
    ProjectSummary: ProjectSummary,
    AwardsComponent: AwardsComponent,
    SlideshowComponent: SlideshowComponent,
    TableComponent: TableComponent,
    ListComponent: ListComponent
  }

export default class ProjectPage extends Component{
  constructor( props ) {
    super( props );

    this.state ={
      pageMeta: require('../components/Project JSONs/' + this.props.fileName + '.json')
    }
  }

  componentDidMount() {
    window.scrollTo(0, 0);
  }

  componentRender( component ) {
    const ComponentName = componentsList[ component.componentName ];
    return <ComponentName source={ component } />
  }
  
  render(){
    const pageMeta = this.state.pageMeta;
    return(
      <MainLayout {...this.props} >
        <div className={ styles.siteWrapper }>
          { pageMeta.components.map( ( component, i ) =>
            <React.Fragment>
              { this.componentRender( component ) }
            </React.Fragment>
          ) }
        </div>
      </MainLayout>
      );
  }
}
